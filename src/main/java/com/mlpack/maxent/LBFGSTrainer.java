package com.mlpack.maxent;

import java.util.Collection;
import java.util.Set;

import org.apache.commons.configuration.Configuration;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.mlpack.Event;
import com.mlpack.EventSpace;
import com.mlpack.Trainer;

public class LBFGSTrainer implements Trainer<MaxentModel> {
    private static Log  log = LogFactory.getLog(LBFGSTrainer.class);

    public LBFGSTrainer() {
        
    }

    public MaxentModel train(EventSpace events, Configuration config) {
        Set<String> featureSet = events.getFeatureSet();
        Set<String> outcomeSet = events.getOutcomeSet();
        double tolerance = config.getDouble("tolerance", 0.000001);
        double smoothing = config.getDouble("smoothing", 0.0);
        int iterations = config.getInt("iterations", 15);

        if (log.isDebugEnabled()) {
            log.debug("Starting L-BFGS iterations...");
            log.debug(String.format("Number of Features: %d", featureSet.size()));
            log.debug(String.format("Number of Outcomes: %d", outcomeSet.size()));
            log.debug(String.format("Tolerance:          %f", tolerance));
            log.debug(String.format("Gaussian Smoothing: %f", smoothing));
        }

        Collection<Event> eList = events.getEvents();
        for (int i = 0; i < iterations; i++) {
            int correct = 0;
            double f = 0.0;

            for (Event event : eList) {
                
            }
        }

        return null;
    }
}
