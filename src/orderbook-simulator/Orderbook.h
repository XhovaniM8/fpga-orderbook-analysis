//
// Created by Xhovani Mali on 3/21/25.
//

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
