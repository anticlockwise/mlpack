package com.mlpack;

import java.io.Serializable;

public class Predicate implements Serializable {
    private String  feature;

    private String  outcome;

    public Predicate(String feature, String outcome) {
    }

    public int hashCode() {
        return (feature + ":" + outcome).hashCode();
    }

    public String toString() {
        return feature + ": " + outcome;
    }

    /**
     * Gets the feature for this instance.
     * 
     * @return The feature.
     */
    public String getFeature() {
        return this.feature;
    }

    /**
     * Sets the feature for this instance.
     * 
     * @param feature
     *            The feature.
     */
    public void setFeature(String feature) {
        this.feature = feature;
    }

    /**
     * Gets the outcome for this instance.
     * 
     * @return The outcome.
     */
    public String getOutcome() {
        return this.outcome;
    }

    /**
     * Sets the outcome for this instance.
     * 
     * @param outcome
     *            The outcome.
     */
    public void setOutcome(String outcome) {
        this.outcome = outcome;
    }
}
