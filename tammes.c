#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
# define M_PI 3.141592653589793
#endif

#define GA 2.39996322972865332

int frame = 0;
int asymmetric = 1;
int N;
double minD = 0;
double bestMinD = 0;

typedef struct {
    double x,y,z;
} vec3;
vec3* pos;
vec3* v;
vec3* best;

double urandom() {
    return 2. * rand() / (RAND_MAX+1.) - 1.;
}

double norm(vec3* v) {
    return sqrt(v->x*v->x+v->y*v->y+v->z*v->z);
}

double norm2(vec3* v) {
    return v->x*v->x+v->y*v->y+v->z*v->z;
}

void normalize(vec3* v) {
    double n = norm(v);
    if (n == 0.) {
        v->x = 1;
        v->y = 0;
        v->z = 0;
    }
    else {
        v->x /= n;
        v->y /= n;
        v->z /= n;
    }
}

double distance(vec3* v1,vec3* v2) {
    double dx = v1->x-v2->x;
    double dy = v1->y-v2->y;
    double dz = v1->z-v2->z;
    return sqrt(dx*dx+dy*dy+dz*dz);
}

double distanceSq(vec3* v1,vec3* v2) {
    double dx = v1->x-v2->x;
    double dy = v1->y-v2->y;
    double dz = v1->z-v2->z;
    return dx*dx+dy*dy+dz*dz;
}

void *safemalloc(long n) {
    void* p = malloc(n);
    if (p == NULL) {
        fprintf(stderr, "Out of memory (requested %ld).\n", n);
        exit(3);
    }
    return p;
}

vec3 add(vec3 a, vec3 b) {
    vec3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

vec3 sub(vec3 a, vec3 b) {
    vec3 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;
    return c;
}

vec3 scale(double a, vec3 b) {
    vec3 c;
    c.x = a * b.x;
    c.y = a * b.y;
    c.z = a * b.z;
    return c;
}

vec3 cross(vec3 a, vec3 b) {
    vec3 c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
    return c;
}

// returns A if the points are not on a triangle
vec3 circumcenter(vec3 A, vec3 B, vec3 C) {
   // https://en.wikipedia.org/wiki/Circumscribed_circle
    vec3 a = sub(A,C);
    vec3 b = sub(B,C);
    vec3 ab = cross(a,b);
    vec3 top = cross(sub(scale(norm2(&a), b), scale(norm2(&b),a)), ab);
    double n2 = norm2(&ab);
    if (n2 == 0) {
        return A;
    }
    else {
        return add(C, scale(0.5/norm2(&ab), top));
    }
}

void closest(vec3* pos, int i, int n, double* d, int* index) {
    int j;
    for (j=0; j<n; j++) {
        d[j] = 3;
        index[j] = -1;
    }
    for (j=0; j<N; j++) {
        double dist;
        if (j != i) {
            dist = distance(&pos[j], &pos[i]);
            int k;
            if (dist < d[n-1]) {
                k = n-1;
                while (k > 0 && dist < d[k-1]) {
                    k--;
                }
                int l;
                for (l=n-1 ; l > k ; l--) {
                    d[l] = d[l-1];
                    index[l] = index[l-1];
                }
                d[k] = dist;
                index[k] = j;
            }
        }
    }
}

void calculateMinD(void) {
    double minD2 = 4;
    int i;

    int N0 = (N % 2 || asymmetric) ? N : N/2;

    for (i=0;i<N0;i++) {
        double d2;
        int j;
        for (j=i+1;j<N;j++) {
            d2 = distanceSq(&pos[i],&pos[j]);
            if (d2 < minD2) minD2 = d2;
        }
    }
    minD = sqrt(minD2);
}

double maxMinD(vec3* pos) {
    double maxMinD2 = -1;
    int i;

//    int N0 = (N % 2) ? N : N/2;

    for (i=0;i<N;i++) {
        double minD2;
        double d2;
        int j;
        minD2 = 4;
        for (j=0;j<N;j++) {
            if (j!=i) {
                d2 = distanceSq(&pos[i],&pos[j]);
                if (d2 < minD2) minD2 = d2;
            }
        }
        if (minD2 > maxMinD2) {
            maxMinD2 = minD2;
        }
    }
    return sqrt(maxMinD2);
}

void update(double approxDx,double p,double minus,double friction) {
    int i;
    int j;
    
    double maxV = 0;
    double thisV;
    
    int N0 = (N % 2 || asymmetric) ? N : N/2;

    vec3* newV = safemalloc(sizeof(vec3) * N0);
    
    for (i=0;i<N0;i++) {
        thisV = norm(&v[i]);
        if (thisV > maxV)
            maxV = thisV;
    }
    
    if (maxV == 0) 
        maxV = 10;
    
    double dt = approxDx / maxV;
    if (dt * friction > 0.75) 
        dt = 0.75 / friction;

#pragma omp parallel
#pragma omp for private(i,j)
    for (i=0;i<N0;i++) {        
        double d;
        double factor;
        newV[i].x = v[i].x * (1. - dt * friction);
        newV[i].y = v[i].y * (1. - dt * friction);
        newV[i].z = v[i].z * (1. - dt * friction);
        
        for (j=0;j<N;j++) {
            if (i==j)
                continue;
            d = distance(&pos[i],&pos[j]);
            if (d == 0) {
                newV[i].x += urandom() * 0.001 * dt;
                newV[i].y += urandom() * 0.001 * dt;
                newV[i].z += urandom() * 0.001 * dt;
                fprintf(stderr, "Collision %d %d: %.9f %.9f %.9f\n", i,j,v[i].x,v[i].y,v[i].z);
                continue;
            }
            factor = dt / pow(d-minus,p+1);
            newV[i].x += factor * (pos[i].x - pos[j].x);
            newV[i].y += factor * (pos[i].y - pos[j].y);
            newV[i].z += factor * (pos[i].z - pos[j].z);
        }
    }
    
    double n;
    for (i=0;i<N0;i++) {
        // Euler-Cromer
        pos[i].x += dt * newV[i].x / 2.;
        pos[i].y += dt * newV[i].y / 2.;
        pos[i].z += dt * newV[i].z / 2.;
        
        n = norm(&pos[i]);
        if (n==0) {
            pos[i].x=1;
            pos[i].y=0;
            pos[i].z=0;
        }
        else {
            pos[i].x /= n;
            pos[i].y /= n;
            pos[i].z /= n;
            v[i] = newV[i];
        }

        if (N0 < N) {
            pos[N0+i].x = -pos[i].x;
            pos[N0+i].y = -pos[i].y;
            pos[N0+i].z = -pos[i].z;
        }
    }
    
    calculateMinD();

    free(newV);
}

double latitude(vec3* v) {
    return 180. / M_PI * atan2(v->z, sqrt(v->x*v->x+v->y*v->y));
}

double longitude(vec3* v) {
    if (v->x == 0 && v->y == 0)
        return 0;
    return 180. / M_PI * atan2(v->x, v->y);
}

void usage(void) {
    fprintf(stderr, "tammes [-animate | -scad] [-repeat repeatCount] [-friction frictionMultiplier] nPoints [nIterations]\n");
}

void dumpFrame(vec3* positions, double minD) {
    int i;
    printf("minD %.9f\n", minD);
    for(i=0;i<N;i++) 
        printf("pos %d %.9f %.9f %.9f\n", i, positions[i].x, positions[i].y, positions[i].z);
    printf("frame %d\n", frame++);
    fflush(stdout);
}

double minDAmongNeighbors(vec3* pos, vec3* base, int* neighbors) {
    int j;
    double closestD = 3;
    for (j=0; j<6; j++) {
        double d = distance(base, &pos[neighbors[j]]);
        if (d<closestD) 
            closestD = d;
    }
    return closestD;
}

void cleanupFrom(vec3* from, vec3* pos, int i, int* neighbors, double eps, double maxMove) {
    vec3 move;
    
    move.x = pos[i].x - from->x;
    move.y = pos[i].y - from->y;
    move.z = pos[i].z - from->z;
    normalize(&move);

    double a;
    
    double myMinD = minDAmongNeighbors(pos, &pos[i], neighbors);
    vec3 adjusted;
    vec3 base = pos[i];
    
    for (a = eps; a <= maxMove; a += eps) {
        adjusted.x = base.x + a * move.x;
        adjusted.y = base.y + a * move.y;
        adjusted.z = base.z + a * move.z;
        normalize(&adjusted);
        double adjMinD = minDAmongNeighbors(pos, &adjusted, neighbors);
        if (adjMinD < myMinD)
            break;
        myMinD = adjMinD;
        pos[i] = adjusted;
    }
}

double closestPointOtherThan(vec3* base, vec3* pos, int omit) {
    double bestD2 = 3;
    int i;
    
    for (i=0; i<N; i++) {
        if (i != omit) {
            double d2 = distanceSq(base, &pos[i]);
            if (d2 < bestD2) {
                bestD2 = d2;
            }
        }
    }

    return sqrt(bestD2);
}

// A bit of greedy local optimization where we try the circumcenters of all triangles
// formed by the six closest neighbors of a given point to see if we can make the
// given point do better.  
void cleanupPointCircumcenter(vec3* pos, int i) {
    double d[6];
    int index[6];
    
    closest(pos, i, 6, d, index);

    vec3 bestPoint = pos[i];
    double bestD = 0;
    double dist;
    
    int a,b,c;
    for (a = 0; a < 6-2; a++) for(b = a+1; b < 6-1 ; b++) for (c = b+1 ; c < 6 ; c++) {
        vec3 circ = circumcenter(pos[index[a]], pos[index[b]], pos[index[c]]);
        normalize(&circ);
        dist = closestPointOtherThan(&circ, pos, i);
        if (dist > bestD) {
            bestPoint = circ;
            bestD = dist;
        }
    }
    
    pos[i] = bestPoint;
}

void cleanup(vec3* pos) {
    if (N < 7)
        return;
    
    int i;
    for (i=0; i<N; i++)
        cleanupPointCircumcenter(pos, i);
}

int
main(int argc, char** argv) {	
	int nIter = 500;
    int repeats = 1;
    int animation = 0;
    int scad = 0;
    int golden = 0;
    double frictionMultiplier = 0.16;
    vec3* bestInRun;
    double bestMinDInRun = 0;
    
    while (argc >= 2 && argv[1][0] == '-') {
        switch(argv[1][1]) {
            case 'g':
                golden = 1;
                break;
            case 'a':
                animation = 1;
                break;
            case 's':
                scad = 1;
                break;
            case 'r':
                if (argc < 3) {
                    usage();
                    return 1;
                }
                repeats = atoi(argv[2]);
                argc--;
                argv++;
                break;
            case 'f':
                if (argc < 3) {
                    usage();
                    return 1;
                }
                frictionMultiplier = atof(argv[2]);
                argc--;
                argv++;
                break;
            default:
                break;
        }
        argc--;
        argv++;
    }
    
    if (argc < 2) {
        usage();
        return 1;
    }
    
    if (golden)
        asymmetric = 1;
    
    N = atoi(argv[1]);
    if (argc >= 3) {
        nIter = atoi(argv[2]);
        if (argc >= 4) 
           frictionMultiplier = atof(argv[3]);
    }
    pos = safemalloc(sizeof(vec3)*N);
    // we waste a bit of memory when N is even, but memory is cheap
    v = safemalloc(sizeof(vec3)*N);
    best = safemalloc(sizeof(vec3)*N);
    bestInRun = safemalloc(sizeof(vec3)*N);
    srand((unsigned int)time(0));
    vec3 origin;
    origin.x = origin.y = origin.z = 0.;
    int i;
    
// impose antipodal symmetry (or almost if N is odd), using idea of https://math.mit.edu/research/highschool/rsi/documents/2012Gautam.pdf
    int N0 = asymmetric ? N : (N+1)/2;

    int r;
    for (r=0;r<repeats;r++) {
        if (golden) {
            for (i=0;i<N;i++) {
                double ratio = (double)(i+1)/(N+1);
                pos[i].x = 2 * sqrt( (1-ratio) * ratio ) * cos(i * GA);
                pos[i].y = 2 * sqrt( (1-ratio) * ratio ) * sin(i * GA);
                pos[i].z = 1-2*ratio;
                v[i].x = 0;
                v[i].y = 0;
                v[i].z = 0;
            }
        }
        else {
            for (i=0;i<N0;i++) {
                double n;
                do {
                    pos[i].x = urandom();
                    pos[i].y = urandom();
                    pos[i].z = urandom();
                    n = norm(&pos[i]);
                } while (n > 1 || n == 0.);
                pos[i].x /= n;
                pos[i].y /= n;
                pos[i].z /= n;
                v[i].x = 0;
                v[i].y = 0;
                v[i].z = 0;
                if (N0+i < N) {
                    pos[N0+i].x = -pos[i].x;
                    pos[N0+i].y = -pos[i].y;
                    pos[N0+i].z = -pos[i].z;
                    v[N0+i].x = 0;
                    v[N0+i].y = 0;
                    v[N0+i].z = 0;
                }
            }
        }
        
        double nextShow = 0;

        calculateMinD();
        if (minD > bestMinD) {
            for (i=0;i<N;i++)
                best[i] = pos[i];
            bestMinD = minD;
        }
        for (i=0;i<N;i++)
            bestInRun[i] = pos[i];
        bestMinDInRun = minD;
        
        if (animation) {
            printf("n %d\n",N);
            dumpFrame(pos,minD);
        }

        for (i=0;i<nIter;i++) {
            double p = 1+i*(8.-1)/nIter;
            if (p>4.5) p=4.5;
            // 7,4.5,3,10,0 : 0.153
            double minus;
            if (p >= 1) {
                minus = 0.9 * minD * i / nIter;
            }
            else {
                minus = 0;
            }
            update(.3*minD+0.00000001, p, minus, frictionMultiplier*N); // 0.0005/N,p); 
            if (minD > bestMinD) {
                int j;
                for(j=0;j<N;j++) {
                    best[j] = pos[j];
                }
                bestMinD = minD;
            }
            if (minD > bestMinDInRun) {
                int j;
                for (j=0;j<N;j++)
                    bestInRun[j] = pos[j];
                bestMinDInRun = minD;
            }
        
            if ((double)i/(nIter-1) >= nextShow || i == nIter-1) {
                fprintf(stderr, "%.0f%% minD=%.5f maxMinD=%.5f bestD=%.5f bestMaxMinD=%.5f p=%.5f       \r", 100.*i/(nIter-1), minD, maxMinD(pos), bestMinD, maxMinD(best), p);
                nextShow += 0.05;
            }
            if (animation) 
                dumpFrame(pos,minD);
        }
        fprintf(stderr, "\n");

        if (N >= 7) {
            for (i=0;i<N;i++)
                pos[i] = bestInRun[i];
            for (i=0;i<70;i++) {
                cleanup(pos);
                calculateMinD();
                if (animation) 
                    dumpFrame(pos,minD);
                fprintf(stderr, "clean(%d) minD=%.5f maxMinD=%.5f     \r", i, minD, maxMinD(pos));
            }
            if (minD > bestMinD) {
                bestMinD = minD;
                for (i=0;i<N;i++)
                    best[i] = pos[i];
            }
            fprintf(stderr, "\n");
        }
    }
    
    fprintf(stderr, "best: minD=%.5f maxMinD=%.5f\n", bestMinD, maxMinD(best));

    if (animation)
        dumpFrame(best,bestMinD);
    else if (scad) {
        puts(
             "diameter=42.7;\n"
             "split=false;\n"
             "horizontalTolerance = 0.3; // for joining halves\n"
             "verticalTolerance = 1.2;\n"
             "chamfer = 0.5;\n");
        
        printf("n=%d;\n"
               "minD=%.9f;\n\n", N, bestMinD);
        //printf("bumpR = 2*sin((1/2)*asin(minD/2));\n");
        printf("points = [");
        for(i=0;i<N;i++) {
            printf("[%.9f,%.9f]", latitude(&best[i]), longitude(&best[i]));
            if (i+1 < N) putchar(',');
        }
        puts ("];\n\n"
              "module dimple() {\n"
              " translate([0,0,1]) sphere(d=minD,$fn=12);\n"
              "}\n\n"
              "module dimples() {\n"
              " union() {\n"
              "  for(i=[0:len(points)-1]) rotate([0,0,points[i][1]]) rotate([90-points[i][0],0,0]) dimple();\n"
              " }\n"
              "}\n\n"
              "module golfball() {\n"
              " render(convexity=2)\n"
              "  scale(diameter/2)\n"
              "   difference() { sphere(r=1,$fn=36); dimples();}\n"
              "}\n\n"
              "module half(upper=true) {\n"
              "  d1 = diameter*.4+2*horizontalTolerance;\n"
              "  render(convexity=1) {\n"
              "   rotate([upper?0:180,0,0]) golfball();\n"
              "   translate([0,0,-diameter*.25-verticalTolerance])\n"
              "    cylinder(d=d1, h=diameter*.5+2*verticalTolerance, $fn=40);\n"
              "   translate([0,0,-0.001])\n"
              "    cylinder(d1=d1+2*chamfer, d2=d1, h=chamfer);\n"
              "  }\n"
              "}\n\n"
              "module halves() {\n"
              "  half();\n"
              "  translate([-8-diameter,0,0]) half();\n"
              "  translate([10+diameter/2,0,0])\n"
              "  render(convexity=0)\n"
              "  intersection() {\n"
              "      cylinder(d=diameter*.4,h=diameter*.5, $fn=40);\n"
              "      cylinder(d1=diameter*.4-2*chamfer+2*diameter*.5,d2=diameter*.4-2*chamfer, h=diameter*.5, $fn=40);\n"
              "  }\n"
              "}\n\n"
              "if (split) halves(); else golfball();\n"
              );
    }
    else {
        for(i=0;i<N;i++) {
            printf("%.9f %.9f %.9f\n", best[i].x, best[i].y, best[i].z);
        }
    }
    
    free(v);
    free(bestInRun);
    free(best);
    free(pos);
    
    return 0;
}
