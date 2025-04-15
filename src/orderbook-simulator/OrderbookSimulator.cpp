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
    double drift = uniform_dist(rng) < 0.6 ? +0.003 : -0.003;
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
    enum EventType { LARGE_BID, LARGE_ASK, CANCEL_BID, CANCEL_ASK, SHIFT_UP, SHIFT_DOWN, SPOOF, SWEEP, NONE };

    if (eventCounts.empty()) {
        eventCounts["LARGE_BID"] = 0;
        eventCounts["LARGE_ASK"] = 0;
        eventCounts["CANCEL_BID"] = 0;
        eventCounts["CANCEL_ASK"] = 0;
        eventCounts["SHIFT_UP"] = 0;
        eventCounts["SHIFT_DOWN"] = 0;
        eventCounts["SPOOF"] = 0;
        eventCounts["SWEEP"] = 0;
        eventCounts["NONE"] = 0;
    }


    std::uniform_int_distribution<int> eventDist(0, 8);
    EventType event = static_cast<EventType>(eventDist(rng));

    auto bidLevels = orderbook.getBidLevels(numLevels);
    auto askLevels = orderbook.getAskLevels(numLevels);

    switch (event) {
        case LARGE_BID:
            eventCounts["LARGE_BID"]++;
            if (!bidLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, std::min(3, (int)bidLevels.size() - 1))(rng);
                double price = bidLevels[level].price;
                double size = bidLevels[level].volume;
                double sizeMultiplier = std::uniform_real_distribution<double>(2.0, 5.0)(rng);
                orderbook.updateBid(price, size * sizeMultiplier);
            }
            break;

        case LARGE_ASK:
            eventCounts["LARGE_ASK"]++;
            if (!askLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, std::min(3, (int)askLevels.size() - 1))(rng);
                double price = askLevels[level].price;
                double size = askLevels[level].volume;
                double sizeMultiplier = std::uniform_real_distribution<double>(2.0, 5.0)(rng);
                orderbook.updateAsk(price, size * sizeMultiplier);
            }
            break;

        case CANCEL_BID:
            eventCounts["CANCEL_BID"]++;
            if (!bidLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, (int)bidLevels.size() - 1)(rng);
                orderbook.updateBid(bidLevels[level].price, bidLevels[level].volume * 0.1);
            }
            break;

        case CANCEL_ASK:
            eventCounts["CANCEL_ASK"]++;
            if (!askLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, (int)askLevels.size() - 1)(rng);
                orderbook.updateAsk(askLevels[level].price, askLevels[level].volume * 0.1);
            }
            break;

        case SHIFT_UP:
            eventCounts["SHIFT_UP"]++;
            currentPrice += tickSize * 3;
            break;

        case SHIFT_DOWN:
            eventCounts["SHIFT_DOWN"]++;
            currentPrice = std::max(currentPrice - tickSize * 3, tickSize);
            break;

        case SPOOF:
            eventCounts["SPOOF"]++;
            if (!askLevels.empty()) {
                int level = std::uniform_int_distribution<int>(0, 2)(rng);
                double spoofPrice = askLevels[level].price;
                double spoofSize = askLevels[level].volume * 10.0;

                // Place spoof order
                orderbook.updateAsk(spoofPrice, spoofSize);

                // Schedule removal after short delay
                std::thread([=, this]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    orderbook.updateAsk(spoofPrice, askLevels[level].volume);  // Revert to original
                }).detach();
            }
            break;

        case SWEEP:
            eventCounts["SWEEP"]++;
            if (!askLevels.empty() && !bidLevels.empty()) {
                bool sweepUp = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.5;

                if (sweepUp) {
                    // Buy-side sweep
                    int sweepDepth = std::uniform_int_distribution<int>(3, 7)(rng);
                    for (int i = 0; i < std::min(sweepDepth, (int)askLevels.size()); ++i) {
                        orderbook.updateAsk(askLevels[i].price, 0.0);
                    }
                    currentPrice += tickSize * sweepDepth * 2.0;
                } else {
                    // Sell-side sweep
                    int sweepDepth = std::uniform_int_distribution<int>(3, 7)(rng);
                    for (int i = 0; i < std::min(sweepDepth, (int)bidLevels.size()); ++i) {
                        orderbook.updateBid(bidLevels[i].price, 0.0);
                    }
                    currentPrice = std::max(currentPrice - tickSize * sweepDepth * 2.0, tickSize);
                }
            }
            break;


        case NONE:
            eventCounts["NONE"]++;
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
    std::cout << "\n--- Random Event Summary ---" << std::endl;
    for (const auto& entry : eventCounts) {
        std::cout << entry.first << ": " << entry.second << std::endl;
    }
}

Orderbook& OrderbookSimulator::getOrderbook() {
    return orderbook;
}
