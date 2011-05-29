package com.mlpack;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

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

    public Set<String> getFeatureNames() {
        HashSet<String> featNames = new HashSet<String>();
        for (Feature feat : features) {
            featNames.add(feat.getName());
        }
        return featNames;
    }

    public Set<Feature> getFeatures() {
        return features;
    }

    public int hashCode() {
        return features.hashCode();
    }
}
