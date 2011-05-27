package com.mlpack;

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
}
