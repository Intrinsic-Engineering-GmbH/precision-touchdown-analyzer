"""Finding aircraft in the recording and measuring where they touched down.

This is the side of the system that is allowed heavy dependencies (OpenCV,
numpy). It only ever *reads* the raw segments the recorder writes, so nothing
here can cost a frame of recording (docs/design.md 1).

    detect.py     MOG2 foreground blobs and a small constant-velocity tracker
    contact.py    the front wheel and its lowest point, tracked over a pass
    touchdown.py  descent / ground-run hinge fit -> sub-frame contact instant
    overlay.py    the proof image: contact frame, target line, measured offset
    pipeline.py   segments in, landing records out
    worker.py     background thread that keeps a session analysed while it records
"""
