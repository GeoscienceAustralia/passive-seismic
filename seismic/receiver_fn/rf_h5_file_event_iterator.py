#!/usr/bin/env python
"""Helper class to iterate over 3-channel event traces in h5 file generated by rf library,
   without loading all the traces into memory. This is a scalable solution for very large files.
"""

import logging

import numpy as np
import h5py
from obspyh5 import dataset2trace
from rf import RFStream

logging.basicConfig()

# pylint: disable=invalid-name, logging-format-interpolation

class IterRfH5FileEvents(object):
    """Helper class to iterate over events in h5 file generated by extract_event_traces.py and pass
       them to RF generator. This class avoids having to load the whole file up front via obspy which
       is slow and not scalable.

       Due to the expected hierarchy structure of the input H5 file, yielded event traces are grouped
       by station ID.

       Data yielded per event can easily be hundreds of kB in size, depending on the length of the
       event traces. rf library defaults to window of (-50, 150) sec about the P wave arrival time.
    """

    def __init__(self, h5_filename, memmap=False):
        self.h5_filename = h5_filename
        self.num_components = 3
        self.memmap_input = memmap

    def _open_source_file(self):
        if self.memmap_input:
            try:
                return h5py.File(self.h5_filename, 'r', driver='core', backing_store=False)
            except OSError as e:
                logger = logging.getLogger(__name__)
                logger.error("Failure to memmap input file with error:\n{}\nReverting to default driver."
                             .format(str(e)))
        # end if
        return h5py.File(self.h5_filename, 'r')
        # end if

    def __iter__(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.info("Scanning jobs metadata from file {}".format(self.h5_filename))
        with self._open_source_file() as f:
            wf_data = f['waveforms']
            num_stations = len(wf_data)
            count = 0
            event_count = 0
            create_event_id = False
            first_loop = True
            for station_id in wf_data:
                count += 1
                logger.info("Station {} {}/{}".format(station_id, count, num_stations))
                station_data = wf_data[station_id]
                for event_time in station_data:
                    event_traces = station_data[event_time]
                    if len(event_traces) != self.num_components:
                        logging.warning("Incorrect number of traces ({}) for stn {} event {}, skipping"
                                        .format(len(event_traces), station_id, event_time))
                        continue

                    if first_loop:
                        first_loop = False
                        tmp = list(event_traces.keys())[0]
                        create_event_id = ('event_id' not in event_traces[tmp].attrs)

                    traces = []
                    skip_trace = False
                    for trace_id in event_traces:
                        trace = dataset2trace(event_traces[trace_id])
                        if np.any(np.isnan(trace.data)):
                            skip_trace = True
                            logging.error("Invalid trace data in {}, skipping".format(trace_id))
                            break
                        traces.append(trace)
                    if skip_trace:
                        continue
                    event_count += 1
                    if create_event_id:
                        event_id = event_count
                    else:
                        event_id = traces[0].stats.event_id
                        assert np.all([(tr.stats.event_id == event_id) for tr in traces])
                    stream = RFStream(traces=traces).sort()
                    yield station_id, event_id, event_time, stream
                # end for
            # end for
        # end with
        logger.info("Yielded {} event traces to process".format(event_count))
    # end func
