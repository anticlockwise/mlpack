package com.mlpack;

public class Counter {
    private double  count   = 0;

    public Counter() {
    }

    public Counter(double count) {
        this.count = count;
    }

    /**
     * Gets the count for this instance.
     * 
     * @return The count.
     */
    public double getCount() {
        return this.count;
    }

    /**
     * Sets the count for this instance.
     * 
     * @param count
     *            The count.
     */
    public void setCount(double count) {
        this.count = count;
    }

    public void addCount(double count) {
        this.count += count;
    }
}
