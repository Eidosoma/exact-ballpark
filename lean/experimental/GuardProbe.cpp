// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// The replay assembles both headers from the pinned Destiny git objects.
#include "movement.hpp"
#include <vector>

using ID = long long;
struct Location { double time; ID collider; };
constexpr int g_collisionMaxIterations = 20;
struct CollisionBallProperties {
    const double m_fuzzyDelta;
    Ball *m_ball;
    double m_tau;
    std::vector<Location> m_locations;
    CollisionBallProperties();
    bool IgnoreCollision(ID, double);
};
#include "methods.hpp"

int main() {
    char mode;
    while (std::cin >> mode) {
        if (mode == 'C') {
            double duration, centre;
            if (!(std::cin >> duration >> centre)) return 1;
            const Vector3d origin(0, 0, 0), end(100000, 0, 0), fixed(centre, 0, 0);
            const double u = CollideTwoSpheres(origin, end, fixed, fixed, 1);
            CollisionBallProperties history;
            history.m_locations = {{0, 7}, {duration, -1}};
            std::printf("%a %d %d\n", u, history.IgnoreCollision(7, u),
                        history.IgnoreCollision(7, u * duration));
        } else if (mode == 'I') {
            double p, v, a, m, k, cached, t, dt;
            if (!(std::cin >> p >> v >> a >> m >> k >> cached >> t >> dt)) return 1;
            Ballpark park;
            park.dt = dt;
            Vector3d pos(p, 0, 0), vel(v, 0, 0);
            park.Integrate(pos, vel, Vector3d(a, 0, 0), m, k, cached, t);
            std::printf("%a %a\n", pos.x, vel.x);
        } else {
            return 1;
        }
    }
    return 0;
}
