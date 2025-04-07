/*
 * Author: Xhovani Mali
 * File: main.cpp
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include "Orderbook.h"
#include "OrderbookSimulator.h"
#include "FeatureExtraction.h"

// Utility function to print timestamp
std::string getTimeString() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// Test orderbook simulation and saving to CSV
void testOrderbookSimulation() {
    std::cout << "[" << getTimeString() << "] Starting orderbook simulation test..." << std::endl;

    OrderbookSimulator simulator(100.0, 0.01, 10, 0.1);
    simulator.runSimulation(10, 100);  // 10 seconds, 100 updates per second

    Orderbook& orderbook = simulator.getOrderbook();
    std::string csvFilename = "orderbook_simulation.csv";
    orderbook.saveHistoryToCSV(csvFilename);
    std::cout << "[" << getTimeString() << "] Saved orderbook history to " << csvFilename << std::endl;
}

// Test full feature extraction pipeline
void testFeatureExtraction() {
    std::cout << "[" << getTimeString() << "] Starting feature extraction test..." << std::endl;

    OrderbookSimulator simulator(100.0, 0.01, 10, 0.1);  // volatility = 0.005
    simulator.runSimulation(30, 100);  // 30 seconds, 100 updates per second

    Orderbook& orderbook = simulator.getOrderbook();
    const auto& states = orderbook.getHistory();  // <-- add getHistory() method if not defined yet

    FeatureExtractor extractor(10, 100.0);
    auto features = extractor.extractFeatures(states);

    // Collect midPrices for labeling
    std::vector<double> midPrices;
    for (const auto& state : states) {
        midPrices.push_back(state.midPrice);
    }

    extractor.prepareLabeledData(features, midPrices, 10, 0.00002);
    extractor.saveToFiles("features.bin", "labels.bin");
}

// Main entry point
int main() {
    testOrderbookSimulation();
    testFeatureExtraction();
    return 0;
}
