#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
# define M_PI 3.141592653589793
#endif

#define GA 2.39996322972865332

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

void normalize(vec3* v) {
    double n = norm(v);
    v->x /= n;
    v->y /= n;
    v->z /= n;
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

void dumpFrame(int frameCount, vec3* positions, double minD) {
    int i;
    printf("minD %.9f\n", minD);
    for(i=0;i<N;i++) 
        printf("pos %d %.9f %.9f %.9f\n", i, positions[i].x, positions[i].y, positions[i].z);
    printf("frame %d\n", frameCount);
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

void cleanupPoint(vec3* pos, int i, double eps) {
    if (N<7)
        return; // we don't usually need this step for small numbers

    double d[6];
    int index[6];
    
    closest(pos, i, 6, d, index);
    if (d[3]-d[0] <= eps)
        return;
    
    if (d[1]-d[0] > eps) {
        cleanupFrom(&pos[index[0]], pos, i, index, eps, d[4]-d[0]);
    }
        
    closest(pos, i, 6, d, index);
    if (d[2]-d[0] > eps) {
        vec3 source;
        source.x = 0.5 * (pos[index[0]].x + pos[index[1]].x);
        source.y = 0.5 * (pos[index[0]].y + pos[index[1]].y);
        source.z = 0.5 * (pos[index[0]].z + pos[index[1]].z);
        cleanupFrom(&source, pos, i, index, eps, d[4]-d[0]);
    }
}

void cleanup(vec3* pos, double eps) {
    if (N<7)
        return; // we don't usually need this step for small numbers
    int i;

    for (i=0;i<N;i++) {
        cleanupPoint(pos, i, eps);
    }
}

int
main(int argc, char** argv) {	
	int nIter = 500;
    int repeats = 1;
    int animation = 0;
    int scad = 0;
    int golden = 0;
    double frictionMultiplier = 0.16;
    
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
    srand(time(0));
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
        if (bestMinD <= minD) {
            for (i=0;i<N;i++)
                best[i] = pos[i];
            bestMinD = minD;
        }
        
        if (animation) {
            printf("n %d\n",N);
            dumpFrame(0,pos,minD);
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
            if ((double)i/(nIter-1) >= nextShow || i == nIter-1) {
                fprintf(stderr, "%.0f%% minD=%.5f maxMinD=%.5f bestD=%.5f bestMaxMinD=%.5f p=%.5f       \r", 100.*i/(nIter-1), minD, maxMinD(pos), bestMinD, maxMinD(best), p);
                nextShow += 0.05;
            }
            if (animation) 
                dumpFrame(i+1,pos,minD);
        }
        fprintf(stderr, "\n");

        if (N >= 7) {
            for (i=0;i<N;i++)
                pos[i] = best[i];
            for (i=0;i<50;i++) {
                cleanup(pos, minD * 0.001 / i);
                calculateMinD();
                if (animation) 
                    dumpFrame(nIter+i,pos,minD);
                fprintf(stderr, "clean(%d) minD=%.5f maxMinD=%.5f     \r", i, minD, maxMinD(pos));
            }
            bestMinD = minD;
            for (i=0;i<N;i++)
                best[i] = pos[i];
            fprintf(stderr, "\n");
        }
    }
    
    if (animation)
        dumpFrame(nIter+N>=7?50+nIter:nIter,best,bestMinD);
    else if (scad) {
        printf("n=%d;\nminD=%.9f;\n", N, bestMinD);
        //printf("bumpR = 2*sin((1/2)*asin(minD/2));\n");
        printf("points = [");
        for(i=0;i<N;i++) {
            printf("[%.9f,%.9f]", latitude(&best[i]), longitude(&best[i]));
            if (i+1 < N) putchar(',');
        }
        puts ("];\n\n");
        puts ("module dimple() {");
        puts (" translate([0,0,1]) sphere(d=minD,$fn=12);");
        puts ("}\n");
        puts ("module dimples() {");
        puts (" union() {");
        puts ("  for(i=[0:len(points)-1]) rotate([0,0,points[i][1]]) rotate([90-points[i][0],0,0]) dimple();");
        puts (" }");
        puts ("}\n");
        puts ("render(convexity=2)");
        puts("difference() {\n sphere(r=1,$fn=36);\n dimples();\n}");
    }
    else {
        for(i=0;i<N;i++) {
            printf("%.9f %.9f %.9f\n", best[i].x, best[i].y, best[i].z);
        }
    }
    
    free(v);
    free(best);
    free(pos);
    
    return 0;
}
