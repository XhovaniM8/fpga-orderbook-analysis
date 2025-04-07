//
// Created by Xhovani Mali on 3/21/25.
//

#include "Orderbook.h"
#include <algorithm>
#include <iostream>
#include <chrono>

Orderbook::Orderbook() = default;

void Orderbook::updateBid(Price price, Volume volume) {
    if (volume > 0) {
        bids[price] = volume;
    } else {
        bids.erase(price);
    }
    history.push_back(getCurrentState());
}

void Orderbook::updateAsk(Price price, Volume volume) {
    if (volume > 0) {
        asks[price] = volume;
    } else {
        asks.erase(price);
    }
    history.push_back(getCurrentState());
}

void Orderbook::clearLevel(bool isBid, Price price) {
    if (isBid) {
        bids.erase(price);
    } else {
        asks.erase(price);
    }
    history.push_back(getCurrentState());
}

std::pair<Price, Volume> Orderbook::getBestBid() const {
    if (bids.empty()) return {0.0, 0.0};
    return *bids.begin();
}

std::pair<Price, Volume> Orderbook::getBestAsk() const {
    if (asks.empty()) return {0.0, 0.0};
    return *asks.begin();
}

Price Orderbook::getMidPrice() const {
    auto bid = getBestBid();
    auto ask = getBestAsk();
    if (bid.first <= 0.0 || ask.first <= 0.0) return 0.0;
    return (bid.first + ask.first) / 2.0;
}

Price Orderbook::getSpread() const {
    auto bid = getBestBid();
    auto ask = getBestAsk();
    if (bid.first <= 0.0 || ask.first <= 0.0) return 0.0;
    return ask.first - bid.first;
}

std::vector<Orderbook::Level> Orderbook::getBidLevels(int depth) const {
    std::vector<Level> levels;
    levels.reserve(depth);
    auto it = bids.begin();
    for (int i = 0; i < depth && it != bids.end(); ++i, ++it) {
        levels.push_back({it->first, it->second});
    }
    return levels;
}

std::vector<Orderbook::Level> Orderbook::getAskLevels(int depth) const {
    std::vector<Level> levels;
    levels.reserve(depth);
    auto it = asks.begin();
    for (int i = 0; i < depth && it != asks.end(); ++i, ++it) {
        levels.push_back({it->first, it->second});
    }
    return levels;
}

Orderbook::State Orderbook::getCurrentState() const {
    auto now = std::chrono::system_clock::now();
    double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count() / 1000.0;

    State state;
    state.timestamp = timestamp;
    state.midPrice = getMidPrice();
    state.spread = getSpread();
    state.bestBid = getBestBid();
    state.bestAsk = getBestAsk();
    state.bidLevels = getBidLevels();
    state.askLevels = getAskLevels();

    return state;
}

void Orderbook::saveHistoryToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // CSV header
    file << "timestamp,mid_price,spread,best_bid_price,best_bid_size,best_ask_price,best_ask_size";
    for (int i = 0; i < 5; ++i) file << ",bid_price_" << i << ",bid_size_" << i;
    for (int i = 0; i < 5; ++i) file << ",ask_price_" << i << ",ask_size_" << i;
    file << '\n';

    for (const auto& state : history) {
        file << state.timestamp << "," << state.midPrice << "," << state.spread << ","
             << state.bestBid.first << "," << state.bestBid.second << ","
             << state.bestAsk.first << "," << state.bestAsk.second;

        for (int i = 0; i < 5; ++i) {
            if (i < state.bidLevels.size()) {
                file << "," << state.bidLevels[i].price << "," << state.bidLevels[i].volume;
            } else {
                file << ",0,0";
            }
        }

        for (int i = 0; i < 5; ++i) {
            if (i < state.askLevels.size()) {
                file << "," << state.askLevels[i].price << "," << state.askLevels[i].volume;
            } else {
                file << ",0,0";
            }
        }

        file << '\n';
    }

    file.close();
}
