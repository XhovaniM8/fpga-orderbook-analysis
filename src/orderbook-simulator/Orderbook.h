/*
 * Author: Xhovani Mali
 * File: Orderbook.h
 *
 * Description:
 * This module implements a simplified limit order book structure.
 * It maintains bid and ask levels as sorted maps, and provides methods to
 * update, clear, and query order book states at multiple price levels.
 *
 * Each update creates a time-stamped snapshot of the book, which includes
 * the top bid/ask levels, spread, and mid-price. These snapshots are stored
 * as a time-series history and can be exported for analysis or used to
 * extract machine learning features.
 *
 * The order book supports real-time simulation and is designed to interact
 * with the OrderbookSimulator and FeatureExtractor components to create
 * labeled training data for FPGA-deployable LSTM networks.
 */

#ifndef ORDERBOOK_ORDERBOOK_H
#define ORDERBOOK_ORDERBOOK_H

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <functional>

using Price = double;
using Volume = double;

class Orderbook {
public:
    struct Level {
        Price price;
        Volume volume;
    };

    struct State {
        double timestamp;
        Price midPrice;
        Price spread;
        std::pair<Price, Volume> bestBid;
        std::pair<Price, Volume> bestAsk;
        std::vector<Level> bidLevels;
        std::vector<Level> askLevels;
    };

    Orderbook();

    // Order updates
    void updateBid(Price price, Volume volume);
    void updateAsk(Price price, Volume volume);
    void clearLevel(bool isBid, Price price);

    // Queries
    std::pair<Price, Volume> getBestBid() const;
    std::pair<Price, Volume> getBestAsk() const;
    Price getMidPrice() const;
    Price getSpread() const;
    std::vector<Level> getBidLevels(int depth = 5) const;
    std::vector<Level> getAskLevels(int depth = 5) const;
    State getCurrentState() const;

    // History
    const std::vector<State>& getHistory() const { return history; }
    void saveHistoryToCSV(const std::string& filename) const;

private:
    std::map<Price, Volume, std::greater<Price>> bids;
    std::map<Price, Volume> asks;
    std::vector<State> history;
};

#endif // ORDERBOOK_ORDERBOOK_H
