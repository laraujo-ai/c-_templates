#include "../../../lib/catch2/catch2amalgamated.hpp"
#include "../include/sort_tracker.hpp"
#include <Eigen/Dense>

using namespace project_x;

TEST_CASE("Tracklet lifecycle", "[tracker]") {
    Eigen::Vector4d bbox;
    bbox << 10, 20, 50, 60;

    GeneralTracklet tracklet(bbox, 0.9, 1);

    REQUIRE(tracklet.age == 0);
    REQUIRE(tracklet.conf == Approx(0.9));

    tracklet.predict();
    REQUIRE(tracklet.age == 1);

    tracklet.update(bbox, 0.95);
    REQUIRE(tracklet.time_since_update == 0);
    REQUIRE(tracklet.hits == 1);
}

TEST_CASE("SortTracker maintains tracks", "[tracker]") {
    SortTracker tracker(3, 1, 0.3);

    std::vector<project_x::Detection> dets = {
        {10, 20, 50, 60, 0.9f, 1}
    };

    auto results1 = tracker.track(dets);
    REQUIRE(results1.size() == 1);
    int64_t id1 = results1[0]["TrackerId"].get<int64_t>();

    dets[0].x1 = 15;
    auto results2 = tracker.track(dets);
    REQUIRE(results2[0]["TrackerId"].get<int64_t>() == id1);
}

TEST_CASE("SortTracker multiple objects", "[tracker]") {
    SortTracker tracker(3, 1, 0.3);

    std::vector<project_x::Detection> dets = {
        {10, 20, 50, 60, 0.9f, 1},
        {100, 100, 150, 150, 0.8f, 2}
    };

    auto results = tracker.track(dets);
    REQUIRE(results.size() == 2);
}
