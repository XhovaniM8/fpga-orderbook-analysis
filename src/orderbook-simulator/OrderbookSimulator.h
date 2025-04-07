//
// Created by Xhovani Mali on 3/21/25.
//

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
};

#endif // ORDERBOOK_ORDERBOOKSIMULATOR_H
