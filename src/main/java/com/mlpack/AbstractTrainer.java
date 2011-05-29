package com.mlpack;

public abstract class AbstractTrainer<T> implements Trainer<T> {
    public AbstractTrainer() {
    }
    /**
     * 
     * 
     * @param featureSet
     * @param len
     * @param probs
     * @return
     */
    protected String eval(FeatureSet featureSet, double[] probs) {
        return "";
    }
}
