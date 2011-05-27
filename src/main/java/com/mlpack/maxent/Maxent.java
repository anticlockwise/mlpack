package com.mlpack.maxent;

import com.mlpack.Event;
import com.mlpack.EventSpace;
import com.mlpack.Feature;
import com.mlpack.FeatureSet;
import java.util.Collection;
import org.apache.commons.configuration.Configuration;

public class Maxent {
    private EventSpace eventSpace = null;

    private Configuration config = null;

    public Maxent(Configuration config) {
        this.config = config;
    }

    public void train() {
        this.train(this.config);
    }

    public void train(Configuration config) {
    }

    public String predict(FeatureSet features) {
        return "";
    }

    public void begin() {
        eventSpace = new EventSpace();
    }

    public void end() {
        int cutoff = config.getInt("cutoff", 1);
        eventSpace.cutEvents(cutoff);
    }

    public void addEvent(FeatureSet features, String outcome, int count) {
    }

    public void addEvent(Collection<Feature> features, String outcome,
            int count) {
        addEvent(new FeatureSet(features), outcome, count);
    }
}
