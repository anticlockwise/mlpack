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

static PyObject *find_class(PyObject *py_module_name, PyObject *py_global_name);

static long pyattr_to_long(PyObject *obj, const char *attr_name);

static double pyattr_to_double(PyObject *obj, const char *attr_name);

static char *pyattr_to_string(PyObject *obj, const char *attr_name);

static void convert_documents(DOC ***docs, double **label, PyObject *py_docs,
        long num_docs, long num_preds);

static void copy_document(WORD *words, double *label, long *queryid,
        long *slackid, double *costfactor, long int *numwords,
        long int max_words_doc, PyObject *py_doc);

static void copy_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
        PyObject *py_learn_parm, PyObject *py_kernel_parm);

static void free_just_model(void *ptr);

static PyObject *svm_light_learn(PyObject *self, PyObject *args);

static PyObject *svm_light_classify(PyObject *self, PyObject *args);

static PyMethodDef PySVMLightMethods[] = {
    {"learn", svm_light_learn, METH_VARARGS,
        "learn(training_data, learn_parameters, kernel_parameters) -> model"},
    {"classify", svm_light_classify, METH_VARARGS,
        "classify(training_data, model) -> list"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initsvmlight(void)
{
    PyObject *module;
    module = Py_InitModule("svmlight", PySVMLightMethods);
    if (module == NULL)
        return;
}

// ===================================================================

static PyObject *find_class(PyObject *py_module_name, PyObject *py_global_name)
{
    PyObject *global = 0, *module;
    module = PySys_GetObject("modules");
    if (module == NULL)
        return NULL;

    module = PyDict_GetItem(module, py_module_name);
    if (module == NULL) {
        module = PyImport_Import(py_module_name);
        if (!module)
            return NULL;
        global = PyObject_GetAttr(module, py_global_name);
        Py_DECREF(module);
    } else {
        global = PyObject_GetAttr(module, py_global_name);
    }
    return global;
}

static void free_just_model(void *ptr)
{
    MODEL *obj = (MODEL *)ptr;
    free_model(obj, 0);
    free(ptr);
}

static PyObject *
svm_light_classify(PyObject *self, PyObject *args)
{
    MODEL *model;
    PyObject *modelobj, *indexer, *result;
    long num_docs, num_preds;
    long docnum = 0, j;
    double dist;

    DOC *doc;
    WORD *words;
    long queryid, slackid, wnum;
    double costfactor, doc_label;

    if (!PyArg_ParseTuple(args, "OO", &modelobj, &indexer)) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (indexer == Py_None || modelobj == Py_None) {
        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject *events = PyObject_CallMethod(indexer, "contexts", NULL);
    num_docs = (long)PyList_Size(events);
    PyObject *plabels = PyObject_CallMethod(indexer, "pred_labels", NULL);
    num_preds = (long)PyList_Size(plabels);

    words = (WORD *)malloc(sizeof(WORD) * (num_preds + 10));
    result = PyList_New(num_docs);

    PyObject *iter = PyObject_GetIter(events);
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        copy_document(words, &doc_label, &queryid, &slackid, &costfactor,
                &wnum, num_preds + 2, item);
        Py_DECREF(item);
    }
}

static PyObject *
svm_light_learn(PyObject *self, PyObject *args)
{
    verbosity = 1;

    // Import the Python SvmModel's module
    PyObject *module = PySys_GetObject("modules");
    PyObject *model_module_name = PyString_FromString("mlpack.svm.model");
    PyObject *model_class_name = PyString_FromString("SvmModel");
    PyObject *svmmodel_class = find_class(model_module_name, model_class_name);
    Py_DECREF(model_module_name);
    Py_DECREF(model_class_name);

    PyObject *indexer        = NULL;
    PyObject *params         = NULL;
    PyObject *kernel_params  = NULL;

    PyObject *result_model   = NULL;

    // Initialization of parameters for the svm_light
    // svm_learn_classication function.
    DOC **docs;
    long num_docs, num_preds, i;
    double *target;
    double *alpha_in = NULL;
    KERNEL_CACHE *kernel_cache;
    LEARN_PARM learn_parm;
    KERNEL_PARM kernel_parm;
    MODEL *model = (MODEL *)my_malloc(sizeof(MODEL));

    if (PyArg_ParseTuple(args, "OOO", &indexer, &params, &kernel_params)) {
        if (indexer == Py_None || params == Py_None || kernel_params == Py_None) {
            Py_INCREF(Py_None);
            return Py_None;
        }

        PyObject *events = PyObject_CallMethod(indexer, "contexts", NULL);
        num_docs = (long)PyList_Size(events);
        PyObject *plabels = PyObject_CallMethod(indexer, "pred_labels", NULL);
        num_preds = (long)PyList_Size(plabels);

        // Copy the parameters from Python object to the svm_light
        // parameters objects.
        copy_parameters(&learn_parm, &kernel_parm, params, kernel_params);
        // Convert from Python Document objects to svm_light docs
        convert_documents(&docs, &target, events, num_docs, num_preds);

        if (kernel_parm.kernel_type == LINEAR) {
            kernel_cache = NULL;
        } else {
            kernel_cache = kernel_cache_init(num_docs, learn_parm.kernel_cache_size);
        }

        if (learn_parm.type == CLASSIFICATION) {
            svm_learn_classification(docs, target, num_docs, num_preds,
                    &learn_parm, &kernel_parm, kernel_cache, model, alpha_in);
        } else if (learn_parm.type == REGRESSION) {
            svm_learn_regression(docs, target, num_docs, num_preds, &learn_parm,
                    &kernel_parm, &kernel_cache, model);
        }

        // TODO: Construct a Python Model object from model
        PyObject *con_args = Py_BuildValue("{s:l,s:l}", "num_docs", num_docs, "num_preds", num_preds);
        if (PyClass_Check(svmmodel_class)) {
            if(!(result_model = PyInstance_New(svmmodel_class, NULL, NULL))) {
                PyErr_Print();
                Py_INCREF(Py_None);
                result_model = Py_None;
            }
        }
        Py_DECREF(svmmodel_class);

        PyObject_SetAttrString(result_model, "num_sv",
                Py_BuildValue("l", model->sv_num));
        PyObject_SetAttrString(result_model, "at_upper_bound",
                Py_BuildValue("l", model->at_upper_bound));
        PyObject_SetAttrString(result_model, "b",
                Py_BuildValue("d", model->b));

        if (model->lin_weights) {
            PyObject *py_linweights = PyList_New((Py_ssize_t) (num_preds+1));
            for (i = 0; i < num_docs+2; i++) {
                PyList_SetItem(py_linweights, (Py_ssize_t)i,
                        Py_BuildValue("d", model->lin_weights[i]));
            }
            PyObject_SetAttrString(result_model, "lin_weights", py_linweights);
        }

        PyObject *py_alpha = PyList_New((Py_ssize_t) (num_docs+2));
        for (i = 0; i < num_docs+2; i++) {
            PyList_SetItem(py_alpha, (Py_ssize_t)i,
                    Py_BuildValue("d", model->alpha[i]));
        }
        PyObject_SetAttrString(result_model, "alpha", py_alpha);

        PyObject *py_index = PyList_New((Py_ssize_t) (num_docs+2));
        for (i = 0; i < num_docs+2; i++) {
            PyList_SetItem(py_index, (Py_ssize_t)i,
                    Py_BuildValue("l", model->index[i]));
        }
        PyObject_SetAttrString(result_model, "index", py_index);

        PyObject_SetAttrString(result_model, "loo_error",
                Py_BuildValue("d", model->loo_error));
        PyObject_SetAttrString(result_model, "loo_recall",
                Py_BuildValue("d", model->loo_recall));
        PyObject_SetAttrString(result_model, "loo_precision",
                Py_BuildValue("d", model->loo_precision));
        PyObject_SetAttrString(result_model, "xa_error",
                Py_BuildValue("d", model->xa_error));
        PyObject_SetAttrString(result_model, "xa_recall",
                Py_BuildValue("d", model->xa_recall));
        PyObject_SetAttrString(result_model, "xa_precision",
                Py_BuildValue("d", model->xa_precision));

        PyObject_SetAttrString(result_model, "maxdiff",
                Py_BuildValue("d", model->maxdiff));

        // Cleaning up
        if (kernel_cache) {
            kernel_cache_cleanup(kernel_cache);
        }
        free(target);

        for (i = 0; i < num_docs; i++) {
            free_example(docs[i], 1);
        }
        free(docs);
        free(alpha_in);
    } else {
        Py_INCREF(Py_None);
        result_model = Py_None;
    }

    return result_model;
}

static void convert_documents(DOC ***docs, double **label, PyObject *py_docs,
        long num_docs, long num_preds)
{
    WORD *words;
    long dnum = 0, wpos, dpos = 0, dneg = 0, dunlab = 0;
    long queryid, slackid, max_docs = num_docs + 2;
    long max_words_doc = num_preds + 2;
    long i;
    double doc_label, costfactor;

    (*docs) = (DOC **)my_malloc(sizeof(DOC *)*max_docs);
    (*label) = (double *)my_malloc(sizeof(double *)*max_docs);

    words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc+10));

    dnum = 0;
    for (i = 0; i < num_docs; i++) {
        PyObject *py_doc = PyList_GetItem(py_docs, (Py_ssize_t)i);
        copy_document(words, &doc_label, &queryid, &slackid, &costfactor,
                &wpos, max_words_doc, py_doc);
        (*label)[dnum] = doc_label;

        if (doc_label > 0)
            dpos++;
        if (doc_label < 0)
            dneg++;
        if (doc_label == 0)
            dunlab++;

        (*docs)[dnum] = create_example(dnum, queryid, slackid, costfactor,
                create_svector(words, "", 1.0));
        dnum++;
    }

    free(words);
}

static void copy_document(WORD *words, double *label, long *queryid,
        long *slackid, double *costfactor, long int *numwords,
        long int max_words_doc, PyObject *py_doc)
{
    PyObject *doc_attrs = PyObject_GetAttrString(py_doc, "attrs");
    (*slackid)    = PyInt_AsLong(PyDict_GetItemString(doc_attrs, "slackid"));
    (*queryid)    = PyInt_AsLong(PyDict_GetItemString(doc_attrs, "queryid"));
    (*costfactor) = PyFloat_AsDouble(PyDict_GetItemString(doc_attrs, "costfactor"));
    (*label)      = pyattr_to_double(py_doc, "oid");

    PyObject *features = PyObject_GetAttrString(py_doc, "context");
    PyObject *fiter    = PyObject_GetIter(features);
    PyObject *feature;
    int wpos = 0;
    while ((feature = PyIter_Next(fiter))) {
        long wnum    = pyattr_to_long(feature, "index");
        double value = pyattr_to_double(feature, "value");
        (words[wpos]).wnum   = wnum;
        (words[wpos]).weight = (FVAL)value;
        wpos++;
        Py_DECREF(feature);
    }
    Py_DECREF(fiter);

    words[wpos].wnum = 0;
    (*numwords) = wpos + 1;
}

static void copy_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm,
        PyObject *py_learn_parm, PyObject *py_kernel_parm)
{
    strcpy (learn_parm->predfile, "trans_predictions");
    strcpy (learn_parm->alphafile, "");
    learn_parm->type                     = pyattr_to_long(py_learn_parm, "svm_type");
    learn_parm->biased_hyperplane        = pyattr_to_long(py_learn_parm, "biased_hyperplane");
    learn_parm->sharedslack              = pyattr_to_long(py_learn_parm, "sharedslack");
    learn_parm->remove_inconsistent      = pyattr_to_long(py_learn_parm, "remove_inconsistent");
    learn_parm->skip_final_opt_check     = !pyattr_to_long(py_learn_parm, "skip_final_opt_check");
    learn_parm->svm_maxqpsize            = pyattr_to_long(py_learn_parm, "maxqpsize");
    learn_parm->svm_newvarsinqp          = pyattr_to_long(py_learn_parm, "newvarsinqp");
    learn_parm->svm_iter_to_shrink       = pyattr_to_long(py_learn_parm, "iter_to_shrink");
    learn_parm->maxiter                  = pyattr_to_long(py_learn_parm, "maxiter");
    learn_parm->kernel_cache_size        = pyattr_to_long(py_learn_parm, "kernel_cache_size");
    learn_parm->svm_c                    = pyattr_to_double(py_learn_parm, "c");
    learn_parm->eps                      = pyattr_to_double(py_learn_parm, "eps");
    learn_parm->transduction_posratio    = pyattr_to_double(py_learn_parm, "transduction_posratio");
    learn_parm->svm_costratio            = pyattr_to_double(py_learn_parm, "costratio");
    learn_parm->svm_costratio_unlab      = pyattr_to_double(py_learn_parm, "costratio_unlab");
    learn_parm->svm_unlabbound           = 1E-5;
    learn_parm->epsilon_crit             = pyattr_to_double(py_learn_parm, "epsilon_crit");
    learn_parm->epsilon_a                = 1E-15;
    learn_parm->compute_loo              = pyattr_to_long(py_learn_parm, "compute_loo");
    learn_parm->rho                      = pyattr_to_double(py_learn_parm, "rho");
    learn_parm->xa_depth                 = pyattr_to_long(py_learn_parm, "xa_depth");
    kernel_parm->kernel_type             = pyattr_to_long(py_kernel_parm, "kernel_type");
    kernel_parm->poly_degree             = pyattr_to_long(py_kernel_parm, "poly_degree");
    kernel_parm->rbf_gamma               = pyattr_to_double(py_kernel_parm, "rbf_gamma");
    kernel_parm->coef_lin                = pyattr_to_double(py_kernel_parm, "coef_lin");
    kernel_parm->coef_const              = pyattr_to_double(py_kernel_parm, "coef_const");
    strcpy(kernel_parm->custom, pyattr_to_string(py_kernel_parm, "custom"));
}

static long pyattr_to_long(PyObject *obj, const char *attr_name)
{
    PyObject *attr = PyObject_GetAttrString(obj, attr_name);
    return PyInt_AsLong(attr);
}

static double pyattr_to_double(PyObject *obj, const char *attr_name)
{
    PyObject *attr = PyObject_GetAttrString(obj, attr_name);
    return PyFloat_AsDouble(attr);
}

static char *pyattr_to_string(PyObject *obj, const char *attr_name)
{
    PyObject *attr = PyObject_GetAttrString(obj, attr_name);
    return PyString_AsString(attr);
}
