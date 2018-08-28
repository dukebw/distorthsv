# distorthsv


## distorthsv tests

1. Build test: run `python3 setup.py build --force install --user --force`.

   Pass criteria: the module should build and install without error.

   Note that your system may require the `DISTORTHSV_EXTRA_LIBS` environment
   variable to be set to `omp`. This is to overcome some discrepancies linking
   OpenMP on different systems, where some systems seem to require the `-lomp`
   flag, while others require this flag _not_ to be set.

   If you need `DISTORTHSV_EXTRA_LIBS=omp` then using the distorthsv module
   will result in an error like this:

   ```python
   ImportError: /export/mlrg/bduke/.local/lib/python3.6/site-packages/distorthsv.cpython-36m-x86_64-linux-gnu.so: undefined symbol: omp_get_thread_num
   ```

   In this case, rebuild like so:
   `DISTORTHSV_EXTRA_LIBS=omp python3 setup.py build --force install --user --force`.

2. Run test:
   `CUDA_VISIBLE_DEVICES= distorthsv_test --filename <video-filename> --width <width> --height <height>`

   Pass criteria: The timed portion of each iteration should take less than 500ms.

   Over 20 iterations, hue, saturation, brightness and contrast are distorted
   one at a time over a period of five iterations each. The video is flipped
   left-right every iteration.
