package com.mlpack;

import java.io.Serializable;

public class Feature implements Serializable {
    private String name;

    private double value;

    public Feature(String name, double value) {
        this.name = name;
        this.value = value;
    }

    public Feature(String name) {
        this(name, 1.0);
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getValue() {
        return value;
    }

    public int hashCode() {
        return name.hashCode();
    }
}
