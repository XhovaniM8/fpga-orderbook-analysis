//
// Created by Xhovani Mali on 3/21/25.
//

#include "FeatureExtraction.h"
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>

// Convert feature struct to flat vector for ML
std::vector<double> OrderbookFeature::toVector() const {
    std::vector<double> vec;

    // Add all features in order
    vec.push_back(priceChange);
    vec.push_back(spread);
    vec.push_back(spreadPct);
    vec.push_back(sizeImbalance);
    vec.push_back(vwmp);
    vec.push_back(vwmpDiff);

    // Add arrays
    for (int i = 0; i < 5; i++) vec.push_back(bidDistances[i]);
    for (int i = 0; i < 5; i++) vec.push_back(askDistances[i]);
    for (int i = 0; i < 5; i++) vec.push_back(bidSizesNorm[i]);
    for (int i = 0; i < 5; i++) vec.push_back(askSizesNorm[i]);

    // Add rolling statistics
    vec.push_back(volatility);
    vec.push_back(priceMom1);
    vec.push_back(priceMom5);
    vec.push_back(priceMom10);
    vec.push_back(priceTrend);
    vec.push_back(spreadTrend);

    return vec;
}

FeatureExtractor::FeatureExtractor(int priceFeatureWindow, double volumeNormalization)
        : priceFeatureWindow(priceFeatureWindow), volumeNormalization(volumeNormalization) {
}

std::vector<OrderbookFeature> FeatureExtractor::extractFeatures(const std::vector<Orderbook::State>& states) {
    std::vector<OrderbookFeature> features;

    // Clear history
    priceHistory.clear();
    priceChangeHistory.clear();
    spreadHistory.clear();

    for (const auto& state : states) {
        features.push_back(extractFeature(state));
    }

    return features;
}

OrderbookFeature FeatureExtractor::extractFeature(const Orderbook::State& state) {
    OrderbookFeature feature;

    // Update history
    if (priceHistory.size() >= priceFeatureWindow) {
        priceHistory.pop_front();
        priceChangeHistory.pop_front();
        spreadHistory.pop_front();
    }

    priceHistory.push_back(state.midPrice);

    // Basic features
    feature.spread = state.spread;
    feature.spreadPct = state.spread / state.midPrice;

    // Price change
    if (priceHistory.size() < 2) {
        feature.priceChange = 0.0;
    } else {
        double prevPrice = *(priceHistory.rbegin() + 1);
        feature.priceChange = (state.midPrice - prevPrice) / prevPrice;
    }
    priceChangeHistory.push_back(feature.priceChange);

    // Order book imbalance
    double totalBidSize = state.bestBid.second;
    double totalAskSize = state.bestAsk.second;
    feature.sizeImbalance = (totalBidSize - totalAskSize) / (totalBidSize + totalAskSize);

    // Volume-weighted mid price
    feature.vwmp = (state.bestBid.first * state.bestAsk.second +
                    state.bestAsk.first * state.bestBid.second) /
                   (state.bestBid.second + state.bestAsk.second);
    feature.vwmpDiff = (feature.vwmp - state.midPrice) / state.midPrice;

    // Price level features
    for (int i = 0; i < 5; i++) {
        // Distance from mid
        if (i < state.bidLevels.size()) {
            feature.bidDistances[i] = (state.midPrice - state.bidLevels[i].price) / state.midPrice;
            feature.bidSizesNorm[i] = state.bidLevels[i].volume / volumeNormalization;
        } else {
            feature.bidDistances[i] = 0.0;
            feature.bidSizesNorm[i] = 0.0;
        }

        if (i < state.askLevels.size()) {
            feature.askDistances[i] = (state.askLevels[i].price - state.midPrice) / state.midPrice;
            feature.askSizesNorm[i] = state.askLevels[i].volume / volumeNormalization;
        } else {
            feature.askDistances[i] = 0.0;
            feature.askSizesNorm[i] = 0.0;
        }
    }

    // Rolling statistics
    if (priceHistory.size() >= priceFeatureWindow) {
        // Volatility (standard deviation of price changes)
        double sumSq = 0.0;
        double mean = std::accumulate(priceChangeHistory.begin(), priceChangeHistory.end(), 0.0) /
                      priceChangeHistory.size();

        for (double change : priceChangeHistory) {
            sumSq += (change - mean) * (change - mean);
        }

        feature.volatility = std::sqrt(sumSq / priceChangeHistory.size());

        // Price momentum
        if (priceHistory.size() >= 11) {
            feature.priceMom1 = (priceHistory.back() - *(priceHistory.rbegin() + 1)) /
                                *(priceHistory.rbegin() + 1);
            feature.priceMom5 = (priceHistory.back() - *(priceHistory.rbegin() + 5)) /
                                *(priceHistory.rbegin() + 5);
            feature.priceMom10 = (priceHistory.back() - *(priceHistory.rbegin() + 10)) /
                                 *(priceHistory.rbegin() + 10);
        } else {
            feature.priceMom1 = feature.priceMom5 = feature.priceMom10 = 0.0;
        }

        // Trend features
        spreadHistory.push_back(state.spread);

        // Price trend (average of price changes)
        feature.priceTrend = std::accumulate(priceChangeHistory.begin(), priceChangeHistory.end(), 0.0) /
                             priceChangeHistory.size();

        // Spread trend
        if (spreadHistory.size() >= 2) {
            std::vector<double> spreadDiffs(spreadHistory.size() - 1);
            for (size_t i = 1; i < spreadHistory.size(); i++) {
                spreadDiffs[i-1] = spreadHistory[i] - spreadHistory[i-1];
            }
            feature.spreadTrend = std::accumulate(spreadDiffs.begin(), spreadDiffs.end(), 0.0) /
                                  spreadDiffs.size();
        } else {
            feature.spreadTrend = 0.0;
        }
    } else {
        // Not enough history
        feature.volatility = 0.0;
        feature.priceMom1 = 0.0;
        feature.priceMom5 = 0.0;
        feature.priceMom10 = 0.0;
        feature.priceTrend = 0.0;
        feature.spreadTrend = 0.0;
    }

    return feature;
}

void FeatureExtractor::prepareLabeledData(const std::vector<OrderbookFeature>& features,
                                          const std::vector<double>& midPrices,
                                          int sequenceLength, double threshold) {
    int horizon = 5;

    if (features.size() <= sequenceLength + horizon) {
        std::cerr << "Not enough data for sequence creation" << std::endl;
        return;
    }

    featureVectors.clear();
    labels.clear();

    // Convert all features to flat vectors
    std::vector<std::vector<double>> allFeatureVecs;
    for (const auto& feature : features) {
        allFeatureVecs.push_back(feature.toVector());
    }

    // Initialize label array with unused flag
    std::vector<int> targetLabels(features.size(), -1);

    // Label each data point based on future price movement
    for (size_t i = sequenceLength; i + horizon < features.size(); ++i) {
        double currentPrice = midPrices[i];
        double futurePrice = midPrices[i + horizon];
        double futureReturn = (futurePrice - currentPrice) / currentPrice;

        // Debug print (every 1000 samples)
        if (i % 1000 == 0) {
            std::cout << "[debug] futureReturn[" << i << "] = " << futureReturn << std::endl;
        }

        if (futureReturn > threshold) {
            targetLabels[i] = 0;  // Up
        } else if (futureReturn < -threshold) {
            targetLabels[i] = 1;  // Down
        } else {
            targetLabels[i] = 2;  // No significant change
        }
    }

    // Build LSTM input sequences with corresponding labels
    for (size_t i = 0; i + sequenceLength + horizon < features.size(); ++i) {
        if (targetLabels[i + sequenceLength] != -1) {
            std::vector<double> sequence;

            for (int j = 0; j < sequenceLength; ++j) {
                const auto& vec = allFeatureVecs[i + j];
                sequence.insert(sequence.end(), vec.begin(), vec.end());
            }

            featureVectors.push_back(sequence);
            labels.push_back(targetLabels[i + sequenceLength]);
        }
    }

    std::cout << "Created " << featureVectors.size() << " sequences with labels" << std::endl;

    // Label distribution breakdown
    int up = 0, down = 0, noChange = 0;
    for (int label : labels) {
        if (label == 0) up++;
        else if (label == 1) down++;
        else noChange++;
    }

    std::cout << "Class distribution: Up=" << up
              << ", Down=" << down
              << ", No Change=" << noChange << std::endl;
}


void FeatureExtractor::saveToFiles(const std::string& featuresPath, const std::string& labelsPath) {
    // Save features
    std::ofstream featFile(featuresPath, std::ios::binary);
    if (!featFile) {
        std::cerr << "Failed to open features file for writing: " << featuresPath << std::endl;
        return;
    }

    // Write number of sequences and vector dimensions
    size_t numSequences = featureVectors.size();
    size_t vectorDimension = featureVectors.empty() ? 0 : featureVectors[0].size();

    featFile.write(reinterpret_cast<const char*>(&numSequences), sizeof(numSequences));
    featFile.write(reinterpret_cast<const char*>(&vectorDimension), sizeof(vectorDimension));

    // Write feature vectors
    for (const auto& vec : featureVectors) {
        featFile.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(double));
    }
    featFile.close();

    // Save labels
    std::ofstream labelFile(labelsPath, std::ios::binary);
    if (!labelFile) {
        std::cerr << "Failed to open labels file for writing: " << labelsPath << std::endl;
        return;
    }

    // Write number of labels
    size_t numLabels = labels.size();
    labelFile.write(reinterpret_cast<const char*>(&numLabels), sizeof(numLabels));

    // Write labels
    labelFile.write(reinterpret_cast<const char*>(labels.data()), labels.size() * sizeof(int));
    labelFile.close();

    std::cout << "Saved " << numSequences << " sequences to " << featuresPath << std::endl;
    std::cout << "Saved " << numLabels << " labels to " << labelsPath << std::endl;
}

// Optional: Add a method to load the saved data
void FeatureExtractor::loadFromFiles(const std::string& featuresPath, const std::string& labelsPath) {
    // Load features
    std::ifstream featFile(featuresPath, std::ios::binary);
    if (!featFile) {
        std::cerr << "Failed to open features file for reading: " << featuresPath << std::endl;
        return;
    }

    // Read number of sequences and vector dimensions
    size_t numSequences, vectorDimension;
    featFile.read(reinterpret_cast<char*>(&numSequences), sizeof(numSequences));
    featFile.read(reinterpret_cast<char*>(&vectorDimension), sizeof(vectorDimension));

    // Resize and read feature vectors
    featureVectors.resize(numSequences);
    for (auto& vec : featureVectors) {
        vec.resize(vectorDimension);
        featFile.read(reinterpret_cast<char*>(vec.data()), vectorDimension * sizeof(double));
    }
    featFile.close();

    // Load labels
    std::ifstream labelFile(labelsPath, std::ios::binary);
    if (!labelFile) {
        std::cerr << "Failed to open labels file for reading: " << labelsPath << std::endl;
        return;
    }

    // Read number of labels
    size_t numLabels;
    labelFile.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));

    // Read labels
    labels.resize(numLabels);
    labelFile.read(reinterpret_cast<char*>(labels.data()), numLabels * sizeof(int));
    labelFile.close();

    std::cout << "Loaded " << numSequences << " sequences from " << featuresPath << std::endl;
    std::cout << "Loaded " << numLabels << " labels from " << labelsPath << std::endl;
}

void FeatureExtractor::printLabelStats() const {
    std::map<int, int> counts;
    for (int label : labels) counts[label]++;

    int total = labels.size();
    std::cout << "Class distribution:\n";
    std::cout << "  Up        = " << counts[0] << " (" << (100.0 * counts[0] / total) << "%)\n";
    std::cout << "  Down      = " << counts[1] << " (" << (100.0 * counts[1] / total) << "%)\n";
    std::cout << "  No Change = " << counts[2] << " (" << (100.0 * counts[2] / total) << "%)\n";
}

