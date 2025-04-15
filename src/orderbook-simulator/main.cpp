/*
 * Author: Xhovani Mali
 * File: main.cpp
 *
 * Description:
 * This is the entry point for the FPGA Order Book Analysis pipeline.
 * It simulates a live order book with realistic market behaviors such as spoofing,
 * sweeps, and aggressive order placements/cancellations. The simulation generates
 * a stream of bid/ask updates, which are then used to extract time-series features
 * and label sequences based on future mid-price movement.
 *
 * These labeled feature sequences are saved for training a quantized LSTM model,
 * which can be deployed on FPGA hardware to enable real-time, low-latency prediction
 * of directional price moves in high-frequency trading environments.
 *
 * Main Tasks:
 *  - Run a 10-second order book simulation and save the output to CSV
 *  - Run a 30-second simulation, extract features, assign labels, and save .bin files
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

    OrderbookSimulator simulator(100.0, 0.05, 10, 0.2);
    simulator.runSimulation(10, 100);  // 10 seconds, 100 updates per second
    Orderbook& orderbook = simulator.getOrderbook();
    std::string csvFilename = "orderbook_simulation.csv";
    orderbook.saveHistoryToCSV(csvFilename);
    std::cout << "[" << getTimeString() << "] Saved orderbook history to " << csvFilename << std::endl;
}

// Test full feature extraction pipeline
void testFeatureExtraction() {
    std::cout << "[" << getTimeString() << "] Starting feature extraction test..." << std::endl;

    OrderbookSimulator simulator(100.0, 0.05, 10, 0.2);  // volatility = 0.005
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

    extractor.prepareLabeledData(features, midPrices, 10, 0.000001);
    std::cout << "Created " << features.size() << " sequences with labels" << std::endl;
    extractor.printLabelStats();  // You can add this helper to count class distribution
    extractor.saveToFiles("features.bin", "labels.bin");
}

// Main entry point
int main() {
    testOrderbookSimulation();
    testFeatureExtraction();
    return 0;
}
