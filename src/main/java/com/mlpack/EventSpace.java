package com.mlpack;

import java.util.HashMap;
import java.util.Map;
import java.util.Collection;
import java.util.Set;

public class EventSpace {
    private Map<String, Event> eventMap = new HashMap<String, Event>();

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
            eventMap.put(uid, evt);
        }
    }

    public void addEvent(FeatureSet features, String outcome, int count) {
        addEvent(new Event(features, outcome, count));
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
