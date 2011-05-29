package com.mlpack;

import java.util.Set;

public class Event {
    private FeatureSet features;

    private String outcome;

    private int count;

    public Event(FeatureSet features, String outcome, int count) {
        this.features = features;
        this.outcome = outcome;
        this.count = count;
    }

    public String getUID() {
        return "" + features.hashCode() + outcome;
    }

    public int hashCode() {
        int hash = features.hashCode();
        return ("" + hash + outcome).hashCode();
    }

    public int getCount() {
        return count;
    }

    public void addCount(int count) {
        this.count += count;
    }

    public String getOutcome() {
        return outcome;
    }

    public Set<String> getFeatureNames() {
        return features.getFeatureNames();
    }

    public FeatureSet getFeatureSet() {
        return features;
    }

    public Set<Feature> getFeatures() {
        return features.getFeatures();
    }
}
