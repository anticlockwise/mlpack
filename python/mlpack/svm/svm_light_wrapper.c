/*
 * =====================================================================================
 *
 *       Filename:  svm_light_wrapper.c
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  06/26/2011 21:05:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Rongzhou Shen (rshen), anticlockwise5@gmail.com
 *        Company:  
 *
 * =====================================================================================
 */

#include <Python.h>
#include <svm_common.h>
#include <svm_learn.h>

static PyObject *
svm_light_learn_classification(PyObject *self, PyObject *args)
{
    PyObject *documents      = NULL;
    PyObject *first_doc      = NULL;
    PyObject *params         = NULL;
    PyObject *kernel_factory = NULL;

    PyObject *result_model   = NULL;

    DOC **docs;
    long num_docs, num_preds, i;
    double *target;
    double *alpha_in = NULL;
    KERNEL_CACHE *kernel_cache;

    if (PyArg_ParseTuple(args, "OOO", &documents, &params, &kernel_factory)) {
        num_docs = (long)PyList_Size(documents);
        first_doc = PyList_GetItem(documents, 0);
        PyObject *np = PyObject_CallMethod(first_doc, "num_preds", NULL);
        num_preds = PyInt_AsLong(np);
    } else {
        Py_INCREF(Py_None);
        result_model = PyNone;
    }

    return result_model;
}
