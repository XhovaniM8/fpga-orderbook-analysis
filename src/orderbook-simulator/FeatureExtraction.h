//
// Created by Xhovani Mali on 3/21/25.
//

#ifndef ORDERBOOK_FEATUREEXTRACTION_H
#define ORDERBOOK_FEATUREEXTRACTION_H

#include "Orderbook.h"
#include <vector>
#include <deque>

struct OrderbookFeature {
    double priceChange;
    double spread;
    double spreadPct;
    double sizeImbalance;
    double vwmp;
    double vwmpDiff;
    double bidDistances[5];
    double askDistances[5];
    double bidSizesNorm[5];
    double askSizesNorm[5];
    double volatility;
    double priceMom1;
    double priceMom5;
    double priceMom10;
    double priceTrend;
    double spreadTrend;

    // For LSTM input, convert to flat array
    std::vector<double> toVector() const;
};

class FeatureExtractor {
public:
    FeatureExtractor(int priceFeatureWindow = 10, double volumeNormalization = 100.0);

    // Extract features from a set of orderbook states
    std::vector<OrderbookFeature> extractFeatures(const std::vector<Orderbook::State>& states);

    // Extract single feature from current state
    OrderbookFeature extractFeature(const Orderbook::State& state);

    // Create labeled data for ML
    void prepareLabeledData(const std::vector<OrderbookFeature>& features,
                            const std::vector<double>& midPrices,
                            int sequenceLength = 10,
                            double threshold = 0.0005);

    // Save features and labels to files
    void saveToFiles(const std::string& featuresPath, const std::string& labelsPath);

    // Load features and labels from files
    void loadFromFiles(const std::string& featuresPath, const std::string& labelsPath);

private:
    int priceFeatureWindow;
    double volumeNormalization;

    // Circular buffer for calculating rolling statistics
    std::deque<double> priceHistory;
    std::deque<double> priceChangeHistory;
    std::deque<double> spreadHistory;

    // For storing feature vectors and labels
    std::vector<std::vector<double>> featureVectors;
    std::vector<int> labels;
};

#endif //ORDERBOOK_FEATUREEXTRACTION_H
