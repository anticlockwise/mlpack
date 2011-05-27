package com.mlpack;

import org.apache.commons.configuration.Configuration;

public interface Trainer<T> {
    public T train(EventSpace events, Configuration config);
}
