package com.mlpack;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Collection;
import java.util.Set;

public class EventSpace {
    private Map<String, Event>  eventMap    = new HashMap<String, Event>();

    private Set<String>        featureSet   = new HashSet<String>();

    private Map<String, Counter> featCount = new HashMap<String, Counter>();

    private Set<String>        outcomeSet   = new HashSet<String>();

    public EventSpace() {
    }

    public EventSpace(Collection<Event> events) {
        eventMap.clear();
        for (Event event : events) {
            addEvent(event);
        }
    }

    public void addEvent(Event evt) {
        String uid = evt.getUID();
        if (eventMap.containsKey(uid)) {
            Event e = eventMap.get(uid);
            e.addCount(evt.getCount());
        } else {
            Set<String> featureNames = evt.getFeatureNames();
            featureSet.addAll(featureNames);
            for (String featName : featureNames) {
                Counter count = null;
                if (!featCount.containsKey(featName)) {
                    count = new Counter();
                    featCount.put(featName, count);
                } else {
                    count = featCount.get(featName);
                }
                count.addCount(1);
            }

            outcomeSet.add(evt.getOutcome());
            eventMap.put(uid, evt);
        }
    }

    public void addEvent(FeatureSet features, String outcome, int count) {
        addEvent(new Event(features, outcome, count));
    }

    public Map<String, Counter> getFeatureCount() {
        return featCount;
    }

    public Set<String> getFeatureSet() {
        return featureSet;
    }

    public Set<String> getOutcomeSet() {
        return outcomeSet;
    }

    public Collection<Event> getEvents() {
        return eventMap.values();
    }

    public void cutEvents(int cutoff) {
        Set<String> keys = eventMap.keySet();
        for (String key : keys) {
            Event event = eventMap.get(key);
            if (event.getCount() < cutoff) {
                eventMap.remove(key);
            }
        }
    }
}
