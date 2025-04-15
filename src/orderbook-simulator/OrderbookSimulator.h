/*
 * Author: Xhovani Mali
 * File: OrderbookSimulator.h
 *
 * Description:
 * This module defines the OrderbookSimulator class, which generates synthetic
 * market activity over time for use in training and evaluating predictive models.
 *
 * The simulator evolves a synthetic order book by applying price drift and volatility,
 * along with randomized microstructure behaviors such as spoofing, large orders,
 * cancellations, directional sweeps, and price shifts.
 *
 * It supports timed simulation runs and produces sequences of order book states
 * suitable for feature extraction and supervised labeling. These outputs are used
 * to train LSTM-based models for real-time financial signal detection on FPGA.
 */


#ifndef ORDERBOOK_ORDERBOOKSIMULATOR_H
#define ORDERBOOK_ORDERBOOKSIMULATOR_H

#include "Orderbook.h"
#include <random>
#include <chrono>

class OrderbookSimulator {
public:
    OrderbookSimulator(double initialPrice = 100.0, double tickSize = 0.01,
                       int levels = 10, double volatility = 0.001);

    void generateUpdate();                // Generate a basic market update
    void simulateRandomEvent();           // Simulate random market anomaly (large order, cancellation)
    void runSimulation(int durationSeconds, int updatesPerSecond); // Run full simulation

    Orderbook& getOrderbook();            // Access current orderbook

private:
    Orderbook orderbook;
    double currentPrice;
    double tickSize;
    int numLevels;
    double volatility;

    std::mt19937 rng;
    std::normal_distribution<double> normalDist;

    std::chrono::time_point<std::chrono::system_clock> lastUpdateTime;

    std::map<std::string, int> eventCounts;

};

#endif // ORDERBOOK_ORDERBOOKSIMULATOR_H
