/**
 * Copyright 2018 Brendan Duke.
 *
 * This file is part of distorthsv.
 *
 * distorthsv is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * distorthsv is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * distorthsv. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Distort values in HSV format.
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include <Python.h>
#include <omp.h>
#include <math.h>

PyDoc_STRVAR(module_doc, "Module for distorting HSV.");

/**
 * NOTE(brendan): Code from tensorflow/core/kernels/adjust_saturation_op.cc.
 */
static void
rgb_to_hsv(float r, float g, float b, float* h, float* s, float* v)
{
        float vv = fmaxf(r, fmaxf(g, b));
        float range = vv - fminf(r, fminf(g, b));
        if (vv > 0) {
                *s = range / vv;
        } else {
                *s = 0;
        }
        float norm = 1.0f / (6.0f * range);
        float hh;
        if (r == vv) {
                hh = norm * (g - b);
        } else if (g == vv) {
                hh = norm * (b - r) + 2.0 / 6.0;
        } else {
                hh = norm * (r - g) + 4.0 / 6.0;
        }
        if (range <= 0.0) {
                hh = 0;
        }
        if (hh < 0.0) {
                hh = hh + 1;
        }
        *v = vv;
        *h = hh;
}

/**
 * Algorithm from wikipedia, https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
 */
static void
hsv_to_rgb(float h, float s, float v, float* r, float* g, float* b)
{
        float c = s * v;
        float m = v - c;
        float dh = h * 6;
        float rr, gg, bb;
        int32_t h_category = (int32_t)(dh);
        float fmodu = dh;
        while (fmodu <= 0) {
                fmodu += 2.0f;
        }
        while (fmodu >= 2.0f) {
                fmodu -= 2.0f;
        }
        float x = c * (1 - fabsf(fmodu - 1));
        switch (h_category) {
        case 0:
                rr = c;
                gg = x;
                bb = 0;
                break;
        case 1:
                rr = x;
                gg = c;
                bb = 0;
                break;
        case 2:
                rr = 0;
                gg = c;
                bb = x;
                break;
        case 3:
                rr = 0;
                gg = x;
                bb = c;
                break;
        case 4:
                rr = x;
                gg = 0;
                bb = c;
                break;
        case 5:
                rr = c;
                gg = 0;
                bb = x;
                break;
        default:
                rr = 0;
                gg = 0;
                bb = 0;
                break;
        }
        *r = rr + m;
        *g = gg + m;
        *b = bb + m;
}

static PyArrayObject *
get_inout_vid_array(PyObject *video_obj)
{
        return (PyArrayObject *)PyArray_FROM_OTF(video_obj,
                                                 NPY_FLOAT,
                                                 (NPY_ARRAY_INOUT_ARRAY |
                                                  NPY_ARRAY_ENSUREARRAY));
}

/**
 * Checks `vid_array` for:
 * 1. Being 4D.
 * 2. Being contiguous.
 * 3. Being aligned.
 *
 * Returns a negative value on error, 0 on success.
 */
static int32_t
check_vid_array(PyArrayObject *vid_array, const char *func_name)
{
        char err_msg[128];
        int32_t num_bytes = snprintf(err_msg,
                                     sizeof(err_msg),
                                     "%s: invalid dimensions, array not "
                                     "contiguous or not aligned.",
                                     func_name);
        assert((num_bytes >= 0) &&
               ((uint32_t)num_bytes < sizeof(err_msg)));

        if ((PyArray_NDIM(vid_array) != 4) ||
            !PyArray_IS_C_CONTIGUOUS(vid_array) ||
            !PyArray_ISALIGNED(vid_array)) {
                PyErr_SetString(PyExc_ValueError, err_msg);
                return -1;
        }

        return 0;
}

static float
clampf01(float val)
{
        return fminf(1.f, fmaxf(0.f, val));
}

/**
 * Transforms the hue, saturation and brightness values of an image, using a
 * maximum number of threads specified by `max_num_threads`.
 */
static PyObject *
distorthsv(PyObject *self, PyObject *args, PyObject *kw)
{
        PyObject *video_obj;
        float hue_factor = 1.0;
        float saturation_factor = 1.0;
        float brightness_factor = 1.0;
        uint32_t max_num_threads = 12;
        static char *kwlist[] = {"video",
                                 "hue_factor",
                                 "saturation_factor",
                                 "brightness_factor",
                                 "max_num_threads",
                                 0};

        /* NOTE(brendan): Stop compiler warnings about unused variable. */
        (void)self;

        if (!PyArg_ParseTupleAndKeywords(args,
                                         kw,
                                         "O!|$fffI:distorthsv",
                                         kwlist,
                                         &PyArray_Type,
                                         &video_obj,
                                         &hue_factor,
                                         &saturation_factor,
                                         &brightness_factor,
                                         &max_num_threads))
                return NULL;

        PyArrayObject *vid_array = get_inout_vid_array(video_obj);
        if (vid_array == NULL)
                return NULL;

        if (check_vid_array(vid_array, "distorthsv") < 0)
                goto clean_up;

        omp_set_num_threads(max_num_threads);

        float *video = PyArray_DATA(vid_array);
        npy_intp *dims = PyArray_DIMS(vid_array);
#pragma omp parallel for default(none) \
        shared(dims, video, hue_factor, saturation_factor, brightness_factor)
        for (uint32_t i = 0;
             i < dims[0]*dims[1]*dims[2]*dims[3];
             i += 3) {
                float h;
                float s;
                float v;
                uint32_t h_index = i + 0;
                uint32_t s_index = i + 1;
                uint32_t v_index = i + 2;

                rgb_to_hsv(video[h_index],
                           video[s_index],
                           video[v_index],
                           &h,
                           &s,
                           &v);

                h = clampf01(h*hue_factor);
                s = clampf01(s*saturation_factor);
                v = clampf01(v*brightness_factor);

                hsv_to_rgb(h,
                           s,
                           v,
                           video + h_index,
                           video + s_index,
                           video + v_index);
        }

        Py_DECREF(vid_array);
        Py_INCREF(Py_None);

        return Py_None;

clean_up:
        Py_DECREF(vid_array);
        return NULL;
}

/**
 * Distorts contrast by `contrast_factor`, by linear interpolating between
 * pixel values and the RGB mean over the entire video clip, i.e. for each
 * pixel x, x becomes contrast_factor*x + (1 - contrast_factor)*mean.
 */
static PyObject *
distort_contrast(PyObject *self, PyObject *args, PyObject *kw)
{
        PyObject *video_obj;
        float contrast_factor = 1.0;
        uint32_t max_num_threads = 12;
        static char *kwlist[] = {"video",
                                 "contrast_factor",
                                 "max_num_threads",
                                 0};

        (void)self;

        if (!PyArg_ParseTupleAndKeywords(args,
                                         kw,
                                         "O!|$fI:distort_contrast",
                                         kwlist,
                                         &PyArray_Type,
                                         &video_obj,
                                         &contrast_factor,
                                         &max_num_threads))
                return NULL;

        PyArrayObject *vid_array = get_inout_vid_array(video_obj);
        if (vid_array == NULL)
                return NULL;

        if (check_vid_array(vid_array, "distort_contrast") < 0)
                goto clean_up;

        omp_set_num_threads(max_num_threads);

        float *video = PyArray_DATA(vid_array);
        npy_intp *dims = PyArray_DIMS(vid_array);
        float r_mean = 0.f;
        float g_mean = 0.f;
        float b_mean = 0.f;
        uint32_t num_pixels = dims[0]*dims[1]*dims[2];
#pragma omp parallel for default(none) shared(dims, video, num_pixels) \
        reduction(+:r_mean, g_mean, b_mean)
        for (uint32_t i = 0;
             i < num_pixels*dims[3];
             i += 3) {
                r_mean += video[i + 0];
                g_mean += video[i + 1];
                b_mean += video[i + 2];
        }

        r_mean /= num_pixels;
        g_mean /= num_pixels;
        b_mean /= num_pixels;

        float mean_scale = 1.f - contrast_factor;
        r_mean *= mean_scale;
        g_mean *= mean_scale;
        b_mean *= mean_scale;

#pragma omp parallel for default(none) \
        shared(dims, video, contrast_factor, r_mean, g_mean, b_mean)
        for (uint32_t i = 0;
             i < dims[0]*dims[1]*dims[2]*dims[3];
             i += 3) {
                video[i + 0] = clampf01(contrast_factor*video[i + 0] +
                                        r_mean);
                video[i + 1] = clampf01(contrast_factor*video[i + 1] +
                                        g_mean);
                video[i + 2] = clampf01(contrast_factor*video[i + 2] +
                                        b_mean);
        }

        Py_DECREF(vid_array);
        Py_INCREF(Py_None);

        return Py_None;

clean_up:
        Py_DECREF(vid_array);
        return NULL;
}

/**
 * Flips a video left-right, in place.
 */
static PyObject *
fliplr(PyObject *self, PyObject *args, PyObject *kw)
{
        PyObject *video_obj;
        uint32_t max_num_threads = 12;
        static char *kwlist[] = {"video", "max_num_threads", 0};

        (void)self;

        if (!PyArg_ParseTupleAndKeywords(args,
                                         kw,
                                         "O!|$I:distort_contrast",
                                         kwlist,
                                         &PyArray_Type,
                                         &video_obj,
                                         &max_num_threads))
                return NULL;

        PyArrayObject *vid_array = get_inout_vid_array(video_obj);
        if (vid_array == NULL)
                return NULL;

        if (check_vid_array(vid_array, "fliplr") < 0)
                goto clean_up;

        omp_set_num_threads(max_num_threads);

        float *video = PyArray_DATA(vid_array);
        npy_intp *dims = PyArray_DIMS(vid_array);
#pragma omp parallel for default(none) shared(dims, video)
        for (uint32_t frame = 0;
             frame < dims[0];
             ++frame) {
		for (uint32_t row = 0;
		     row < dims[1];
		     ++row) {
			uint32_t col_start = (frame*dims[1]*dims[2]*dims[3] +
					      row*dims[2]*dims[3]);
			for (uint32_t col = 0;
			     col < dims[2]/2;
			     ++col) {
				float temp_r;
				float temp_g;
				float temp_b;
				uint32_t next_pixel = col_start + col*dims[3];
				temp_r = video[next_pixel + 0];
				temp_g = video[next_pixel + 1];
				temp_b = video[next_pixel + 2];

				uint32_t swap_pixel = (col_start +
						       (dims[2]*dims[3] -
							(col + 1)*dims[3]));
				video[next_pixel + 0] = video[swap_pixel + 0];
				video[next_pixel + 1] = video[swap_pixel + 1];
				video[next_pixel + 2] = video[swap_pixel + 2];

				video[swap_pixel + 0] = temp_r;
				video[swap_pixel + 1] = temp_g;
				video[swap_pixel + 2] = temp_b;
			}
		}
	}

        Py_DECREF(vid_array);
        Py_INCREF(Py_None);

        return Py_None;

clean_up:
        Py_DECREF(vid_array);
        return NULL;
}

static PyMethodDef distorthsv_methods[] = {
        {"distorthsv",
         (PyCFunction)distorthsv,
         METH_VARARGS | METH_KEYWORDS,
         PyDoc_STR("distorthsv(video, hue_factor, saturation_factor, "
                   "brightness_factor, max_num_threads) -> "
                   "output with distorted HSV.")},
        {"distort_contrast",
         (PyCFunction)distort_contrast,
         METH_VARARGS | METH_KEYWORDS,
         PyDoc_STR("distorthsv(video, contrast_factor, max_num_threads) -> "
                   "output with distorted contrast.")},
        {"fliplr",
         (PyCFunction)fliplr,
         METH_VARARGS | METH_KEYWORDS,
         PyDoc_STR("fliplr(video, max_num_threads) -> "
                   "output flipped left-right.")},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef
distorthsvmodule = {
        PyModuleDef_HEAD_INIT,
        "_distorthsv",
        module_doc,
        0,
        distorthsv_methods,
        NULL,
        NULL,
        NULL,
        NULL
};

PyMODINIT_FUNC
PyInit__distorthsv(void)
{
        PyObject *module = PyModuleDef_Init(&distorthsvmodule);
        import_array();

        return module;
}
