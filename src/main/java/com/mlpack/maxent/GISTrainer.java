package com.mlpack.maxent;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.configuration.Configuration;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.mlpack.Counter;
import com.mlpack.Event;
import com.mlpack.EventSpace;
import com.mlpack.Feature;
import com.mlpack.FeatureSet;
import com.mlpack.Predicate;
import com.mlpack.Prior;
import com.mlpack.Trainer;
import com.mlpack.UniformPrior;

public class GISTrainer implements Trainer<MaxentModel> {
    public static final double      NEAR_ZERO            = 0.01;

    public static final double      LL_THRESHOLD         = 0.0001;

    private static Log              log                  =
                                                                 LogFactory
                                                                         .getLog(GISTrainer.class);

    private boolean                 useSimpleSmoothing   = false;

    private boolean                 useSlackParameter    = false;

    private boolean                 useGaussianSmoothing    = false;

    private double                  sigma                = 2.0;

    private double                  smoothingObservation    = 0.1;

    private double                  cfObservedExpect;

    private Map<Predicate, Counter> predCount            =
                                                                 new HashMap<Predicate, Counter>();

    private Map<String, Counter>    featCount            = null;

    private Collection<Event>       eventList            = null;

    private Set<String>             outcomeSet           = null;

    private Map<String, Parameters> params               =
                                                                 new HashMap<String, Parameters>();

    private Map<String, Parameters> modelExpects         =
                                                                 new HashMap<String, Parameters>();

    private Map<String, Parameters> observedExpects      =
                                                                 new HashMap<String, Parameters>();

    private Prior prior = null;

    public MaxentModel train(EventSpace events, Configuration config) {
        if (log.isDebugEnabled()) {
        }

        int cutoff = config.getInt("cutoff", 1);
        int iterations = config.getInt("iterations", 15);

        eventList = events.getEvents();
        featCount = events.getFeatureCount();
        outcomeSet = events.getOutcomeSet();
        prior = new UniformPrior(outcomeSet);
        Set<String> featNames = events.getFeatureSet();

        int correctionConstant = 1;
        for (Event event : eventList) {
            double featSum = 0.0;
            Set<Feature> features = event.getFeatures();
            for (Feature feat : features) {
                featSum += feat.getValue();
                Predicate pred =
                        new Predicate(feat.getName(), event.getOutcome());
                Counter count = null;
                if (!predCount.containsKey(pred)) {
                    count = new Counter();
                    predCount.put(pred, count);
                } else {
                    count = predCount.get(pred);
                }
                count.addCount(event.getCount() * feat.getValue());
            }
            if (featSum > correctionConstant) {
                correctionConstant = (int) Math.ceil(featSum);
            }
        }

        HashSet<String> activeOutcomes = null;
        for (String featName : featNames) {
            if (useSimpleSmoothing) {
                activeOutcomes = outcomeSet;
            } else {
                activeOutcomes = new HashSet<String>();
                for (String outcome : outcomeSet) {
                    Predicate pred = new Predicate(featName, outcome);
                    if (predCount.containsKey(pred)
                            && predCount.get(pred).getCount() > 0
                            && featCount.containsKey(featName)
                            && featCount.get(featName).getCount() >= cutoff) {
                        activeOutcomes.add(outcome);
                    }
                }
            }

            params.put(featName, new Parameters());
            modelExpects.put(featName, new Parameters());
            observedExpects.put(featName, new Parameters());

            for (String outcome : activeOutcomes) {
                params.get(featName).setParameter(outcome, 0.0);
                modelExpects.get(featName).setParameter(outcome, 0.0);
                Predicate pred = new Predicate(featName, outcome);
                if (predCount.containsKey(pred)
                        && predCount.get(pred).getCount() > 0) {
                    observedExpects.get(featName).setParameter(outcome,
                            predCount.get(pred).getCount());
                } else if (useSimpleSmoothing) {
                    observedExpects.get(featName).setParameter(outcome,
                            smoothingObservation);
                }
            }
        }

        if (useSlackParameter) {
            int cfvalSum = 0;
            for (Event event : eventList) {
                for (String feat : event.getFeatureNames()) {
                    if (!modelExpects.get(feat).contains(event.getOutcome())) {
                        cfvalSum += event.getCount();
                    }
                }
                cfvalSum +=
                        (correctionConstant - event.getFeatureNames().size())
                                * event.getCount();
            }

            if (cfvalSum == 0) {
                cfObservedExpect = Math.log(NEAR_ZERO);
            } else {
                cfObservedExpect = Math.log(cvfalSum);
            }
        }

        predCount = null;
        findParameters(iterations, correctionConstant);

        return null;
    }

    private void findParameters(int iterations, int correctionConstant) {
        double preLL = 0.0; // Likelihood of previous iteration;
        double curLL = 0.0; // Likelihood of current iteration;
        log.info(String.foramt("Performing %d iterations", iterations));
        for (int i = 0; i < iterations; i++) {
            curLL = nextIteration(correctionConstant);
            if (i > 1) {
                if (preLL > curLL) {
                    log.info("Model diverging: log likelihood decreased");
                    break;
                }
                if (curLL - preLL < LL_THRESHOLD) {
                    break;
                }
            }
            preLL = curLL;
        }
    }

    private double nextIteration(int correctionConstant) {
        Map<String, Counter> modelDistribution = new HashMap<String, Counter>();
        double loglikelihood = 0.0;
        for (Event event : eventList) {
            prior.logPrior(modelDistribution, event.getFeatureSet());
            MaxentModel.eval(event.getFeatureSet(), modelDistribution, params);

            FeatureSet featureSet = event.getFeatureSet();
        }
        return loglikelihood;
    }
}
