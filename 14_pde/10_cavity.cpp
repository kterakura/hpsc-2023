#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>

const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

void buildMesh(std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < nx; ++i) {
        x.push_back(i * dx);
    }
    for (int i = 0; i < ny; ++i) {
        y.push_back(i * dy);
    }
}

void initializeArrays(std::vector<std::vector<double> >& u,
                      std::vector<std::vector<double> >& v,
                      std::vector<std::vector<double> >& p,
                      std::vector<std::vector<double> >& b) {
    u.resize(ny, std::vector<double>(nx, 0.0));
    v.resize(ny, std::vector<double>(nx, 0.0));
    p.resize(ny, std::vector<double>(nx, 0.0));
    b.resize(ny, std::vector<double>(nx, 0.0));
}

void solve(std::vector<std::vector<double> >& u,
           std::vector<std::vector<double> >& v,
           std::vector<std::vector<double> >& p,
           std::vector<std::vector<double> >& b,
           const std::vector<double>& x,
           const std::vector<double>& y) {
    for (int n = 0; n < nt; ++n) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                b[j][i] = rho * (1.0 / dt * 
                                ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
                                ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) * ((u[j][i + 1] - u[j][i - 1]) / (2 * dx)) - 2.0 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy)) * 
                                ((v[j][i + 1] - v[j][i - 1]) / (2 * dx)) - ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)) * ((v[j + 1][i] - v[j - 1][i]) / (2 * dy)));
            }
        }

        for (int it = 0; it < nit; ++it) {
            std::vector<std::vector<double> > pn = p;
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    p[j][i] =
                        (dy * dy * (pn[j][i + 1] + pn[j][i - 1]) +
                         dx * dx * (pn[j + 1][i] + pn[j - 1][i]) -
                         b[j][i] * dx * dx * dy * dy) /
                        (2.0 * (dx * dx + dy * dy));
                }
            }
            for (int i = 0; i < nx; ++i) {
                p[0][i] = p[1][i];
                p[ny - 1][i] = 0.0;
            }
            for (int j = 0; j < ny; ++j) {
                p[j][0] = p[j][1];
                p[j][nx - 1] = p[j][nx - 2];
            }
        }

        std::vector<std::vector<double> > un = u;
        std::vector<std::vector<double> > vn = v;
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                          vn[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                          dt / (2.0 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                          nu * dt / (dx * dx) * (un[j][i + 1] - 2.0 * un[j][i] + un[j][i - 1]) +
                          nu * dt / (dy * dy) * (un[j + 1][i] - 2.0 * un[j][i] + un[j - 1][i]);

                v[j][i] = vn[j][i] - un[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                          vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                          dt / (2.0 * rho * dy) * (p[j + 1][i] - p[j - 1][i]) +
                          nu * dt / (dx * dx) * (vn[j][i + 1] - 2.0 * vn[j][i] + vn[j][i - 1]) +
                          nu * dt / (dy * dy) * (vn[j + 1][i] - 2.0 * vn[j][i] + vn[j - 1][i]);
            }
        }

        for (int i = 0; i < nx; ++i) {
            u[0][i] = 0.0;
            u[ny - 1][i] = 1.0;
            v[0][i] = 0.0;
            v[ny - 1][i] = 0.0;
        }

        for (int j = 0; j < ny; ++j) {
            u[j][0] = 0.0;
            u[j][nx - 1] = 0.0;
            v[j][0] = 0.0;
            v[j][nx - 1] = 0.0;
        }

        // Plotting (using file output instead of matplotlib)
        std::ofstream outFile("output/" + std::to_string(n) + ".txt");
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                outFile << x[i] << " " << y[j] << " " << p[j][i] << " " << u[j][i] << " " << v[j][i] << std::endl;
            }
        }
        outFile.close();
    }
}

int main() {
    std::vector<double> x, y;
    buildMesh(x, y);

    std::vector<std::vector<double> > u, v, p, b;
    initializeArrays(u, v, p, b);

    solve(u, v, p, b, x, y);

    return 0;
}
