package com.mlpack;

import java.util.*;

public class FeatureSet implements Iterable<Feature> {
    private HashSet<Feature> features = new HashSet<Feature>();

    public FeatureSet() {
    }

    public FeatureSet(Collection<Feature> features) {
        this.features.addAll(features);
    }

    public Iterator<Feature> iterator() {
        return features.iterator();
    }

    public void addFeature(Feature feature) {
        features.add(feature);
    }

    public int hashCode() {
        return features.hashCode();
    }
}
