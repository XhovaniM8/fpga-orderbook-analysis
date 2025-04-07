/*
 * Created by Xhovani Mali on 3/21/25.
 */

#include "OrderbookSimulator.h"
#include <thread>
#include <iostream>
#include <cmath>

OrderbookSimulator::OrderbookSimulator(double initialPrice, double tickSize,
                                       int levels, double volatility)
        : currentPrice(initialPrice),
          tickSize(tickSize),
          numLevels(levels),
          volatility(volatility),
          normalDist(0.0, volatility) {
    std::random_device rd;
    rng.seed(rd());

    lastUpdateTime = std::chrono::system_clock::now();

    for (int i = 1; i <= numLevels; ++i) {
        double bidPrice = currentPrice - i * tickSize;
        double askPrice = currentPrice + i * tickSize;

        double baseBidSize = 10.0 * (1.0 + 0.5 * static_cast<double>(rand()) / RAND_MAX) / (1.0 + 0.2 * i);
        double baseAskSize = 10.0 * (1.0 + 0.5 * static_cast<double>(rand()) / RAND_MAX) / (1.0 + 0.2 * i);

        orderbook.updateBid(bidPrice, baseBidSize);
        orderbook.updateAsk(askPrice, baseAskSize);
    }
}

void OrderbookSimulator::generateUpdate() {
    auto currentTime = std::chrono::system_clock::now();
    double timeDelta = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastUpdateTime).count() / 1000.0;

    lastUpdateTime = currentTime;

    // Changed this
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    double drift = uniform_dist(rng) < 0.5 ? +0.001 : -0.005;
    double noise = normalDist(rng) * std::sqrt(timeDelta);
    double priceChange = drift + noise;
    currentPrice = std::max(currentPrice + priceChange, tickSize);

    for (int i = 1; i <= numLevels; ++i) {
        double bidPrice = currentPrice - i * tickSize;
        double askPrice = currentPrice + i * tickSize;

        double baseBidSize = 10.0 * (1.0 + 0.5 * static_cast<double>(rand()) / RAND_MAX) / (1.0 + 0.2 * i);
        double baseAskSize = 10.0 * (1.0 + 0.5 * static_cast<double>(rand()) / RAND_MAX) / (1.0 + 0.2 * i);

        orderbook.updateBid(bidPrice, baseBidSize);
        orderbook.updateAsk(askPrice, baseAskSize);
    }
}

void OrderbookSimulator::simulateRandomEvent() {
    enum EventType { LARGE_BID, LARGE_ASK, CANCEL_BID, CANCEL_ASK, SHIFT_UP, SHIFT_DOWN, NONE };

    std::uniform_int_distribution<int> eventDist(0, 6);
    EventType event = static_cast<EventType>(eventDist(rng));

    auto bidLevels = orderbook.getBidLevels(numLevels);
    auto askLevels = orderbook.getAskLevels(numLevels);

    switch (event) {
        case LARGE_BID:
            if (!bidLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, std::min(3, (int)bidLevels.size() - 1))(rng);
                double price = bidLevels[level].price;
                double size = bidLevels[level].volume;
                double sizeMultiplier = std::uniform_real_distribution<double>(2.0, 5.0)(rng);
                orderbook.updateBid(price, size * sizeMultiplier);
            }
            break;

        case LARGE_ASK:
            if (!askLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, std::min(3, (int)askLevels.size() - 1))(rng);
                double price = askLevels[level].price;
                double size = askLevels[level].volume;
                double sizeMultiplier = std::uniform_real_distribution<double>(2.0, 5.0)(rng);
                orderbook.updateAsk(price, size * sizeMultiplier);
            }
            break;

        case CANCEL_BID:
            if (!bidLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, (int)bidLevels.size() - 1)(rng);
                orderbook.updateBid(bidLevels[level].price, bidLevels[level].volume * 0.1);
            }
            break;

        case CANCEL_ASK:
            if (!askLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, (int)askLevels.size() - 1)(rng);
                orderbook.updateAsk(askLevels[level].price, askLevels[level].volume * 0.1);
            }
            break;

        case SHIFT_UP:
            currentPrice += tickSize * 3;
            break;

        case SHIFT_DOWN:
            currentPrice = std::max(currentPrice - tickSize * 3, tickSize);
            break;

        case NONE:
            break;
    }
}

void OrderbookSimulator::runSimulation(int durationSeconds, int updatesPerSecond) {
    std::cout << "Starting orderbook simulation for " << durationSeconds << " seconds..." << std::endl;

    auto startTime = std::chrono::system_clock::now();
    auto updateInterval = std::chrono::milliseconds(1000 / updatesPerSecond);
    auto nextUpdateTime = startTime;

    int updateCount = 0;

    while (std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - startTime).count() < durationSeconds) {

        auto currentTime = std::chrono::system_clock::now();

        if (currentTime >= nextUpdateTime) {
            generateUpdate();

            if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.2) {
                simulateRandomEvent();
            }

            ++updateCount;
            if (updateCount % 100 == 0) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        currentTime - startTime).count();
                std::cout << "Processed " << updateCount << " updates, time elapsed: "
                          << elapsed << "s" << std::endl;
            }

            nextUpdateTime += updateInterval;
            if (nextUpdateTime < currentTime) {
                nextUpdateTime = currentTime + updateInterval;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Simulation complete. Generated " << updateCount << " orderbook updates." << std::endl;
}

Orderbook& OrderbookSimulator::getOrderbook() {
    return orderbook;
}
