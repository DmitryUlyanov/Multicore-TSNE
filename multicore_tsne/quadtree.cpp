/*
 *  quadtree.cpp
 *  Implementation of a quadtree in two dimensions + Barnes-Hut algorithm for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "quadtree.h"



// Checks whether a point lies in a cell
bool Cell::containsPoint(double point[])
{
    if (x - hw > point[0]) return false;
    if (x + hw < point[0]) return false;
    if (y - hh > point[1]) return false;
    if (y + hh < point[1]) return false;
    return true;
}


// Default constructor for quadtree -- build tree, too!
QuadTree::QuadTree(double* inp_data, int N)
{

    // Compute mean, width, and height of current map (boundaries of quadtree)
    double* mean_Y = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++) mean_Y[d] = .0;
    double*  min_Y = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++)  min_Y[d] =  DBL_MAX;
    double*  max_Y = new double[QT_NO_DIMS]; for (int d = 0; d < QT_NO_DIMS; d++)  max_Y[d] = -DBL_MAX;
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < QT_NO_DIMS; d++) {
            mean_Y[d] += inp_data[n * QT_NO_DIMS + d];
            if (inp_data[n * QT_NO_DIMS + d] < min_Y[d]) min_Y[d] = inp_data[n * QT_NO_DIMS + d];
            if (inp_data[n * QT_NO_DIMS + d] > max_Y[d]) max_Y[d] = inp_data[n * QT_NO_DIMS + d];
        }
    }
    for (int d = 0; d < QT_NO_DIMS; d++) mean_Y[d] /= (double) N;

    // Construct quadtree
    init(NULL, inp_data, mean_Y[0], mean_Y[1], max(max_Y[0] - mean_Y[0], mean_Y[0] - min_Y[0]) + 1e-5,
                                               max(max_Y[1] - mean_Y[1], mean_Y[1] - min_Y[1]) + 1e-5);
    fill(N);
    delete[] mean_Y; delete[] max_Y; delete[] min_Y;
}


// Constructor for quadtree with particular size and parent -- build the tree, too!
QuadTree::QuadTree(double* inp_data, int N, double inp_x, double inp_y, double inp_hw, double inp_hh)
{
    init(NULL, inp_data, inp_x, inp_y, inp_hw, inp_hh);
    fill(N);
}

// Constructor for quadtree with particular size and parent -- build the tree, too!
QuadTree::QuadTree(QuadTree* inp_parent, double* inp_data, int N, double inp_x, double inp_y, double inp_hw, double inp_hh)
{
    init(inp_parent, inp_data, inp_x, inp_y, inp_hw, inp_hh);
    fill(N);
}


// Constructor for quadtree with particular size (do not fill the tree)
QuadTree::QuadTree(double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh)
{
    init(NULL, inp_data, inp_x, inp_y, inp_hw, inp_hh);
}


// Constructor for quadtree with particular size and parent (do not fill the tree)
QuadTree::QuadTree(QuadTree* inp_parent, double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh)
{
    init(inp_parent, inp_data, inp_x, inp_y, inp_hw, inp_hh);
}


// Main initialization function
void QuadTree::init(QuadTree* inp_parent, double* inp_data, double inp_x, double inp_y, double inp_hw, double inp_hh)
{
    parent = inp_parent;
    data = inp_data;
    is_leaf = true;
    size = 0;
    cum_size = 0;
    boundary.x  = inp_x;
    boundary.y  = inp_y;
    boundary.hw = inp_hw;
    boundary.hh = inp_hh;
    northWest = NULL;
    northEast = NULL;
    southWest = NULL;
    southEast = NULL;
    for (int i = 0; i < QT_NO_DIMS; i++) {
        center_of_mass[i] = .0;
    }
}


// Destructor for quadtree
QuadTree::~QuadTree()
{
    delete northWest;
    delete northEast;
    delete southWest;
    delete southEast;
}


// Update the data underlying this tree
void QuadTree::setData(double* inp_data)
{
    data = inp_data;
}


// Get the parent of the current tree
QuadTree* QuadTree::getParent()
{
    return parent;
}


// Insert a point into the QuadTree
bool QuadTree::insert(int new_index)
{
    // Ignore objects which do not belong in this quad tree
    double* point = data + new_index * QT_NO_DIMS;
    if (!boundary.containsPoint(point))
        return false;

    // Online update of cumulative size and center-of-mass
    cum_size++;
    double mult1 = (double) (cum_size - 1) / (double) cum_size;
    double mult2 = 1.0 / (double) cum_size;
    for (int d = 0; d < QT_NO_DIMS; d++) {
        center_of_mass[d] = center_of_mass[d] * mult1 + mult2 * point[d];
    }

    // If there is space in this quad tree and it is a leaf, add the object here
    if (is_leaf && size < QT_NODE_CAPACITY) {
        index[size] = new_index;
        size++;
        return true;
    }

    // Don't add duplicates for now (this is not very nice)
    bool any_duplicate = false;
    for (int n = 0; n < size; n++) {
        bool duplicate = true;
        for (int d = 0; d < QT_NO_DIMS; d++) {
            if (point[d] != data[index[n] * QT_NO_DIMS + d]) { duplicate = false; break; }
        }
        any_duplicate = any_duplicate | duplicate;
    }
    if (any_duplicate) return true;

    // Otherwise, we need to subdivide the current cell
    if (is_leaf) subdivide();

    // Find out where the point can be inserted
    if (northWest->insert(new_index)) return true;
    if (northEast->insert(new_index)) return true;
    if (southWest->insert(new_index)) return true;
    if (southEast->insert(new_index)) return true;

    // Otherwise, the point cannot be inserted (this should never happen)
    return false;
}


// Create four children which fully divide this cell into four quads of equal area
void QuadTree::subdivide() {

    // Create four children
    northWest = new QuadTree(this, data, boundary.x - .5 * boundary.hw, boundary.y - .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
    northEast = new QuadTree(this, data, boundary.x + .5 * boundary.hw, boundary.y - .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
    southWest = new QuadTree(this, data, boundary.x - .5 * boundary.hw, boundary.y + .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);
    southEast = new QuadTree(this, data, boundary.x + .5 * boundary.hw, boundary.y + .5 * boundary.hh, .5 * boundary.hw, .5 * boundary.hh);

    // Move existing points to correct children
    for (int i = 0; i < size; i++) {
        bool success = false;
        if (!success) success = northWest->insert(index[i]);
        if (!success) success = northEast->insert(index[i]);
        if (!success) success = southWest->insert(index[i]);
        if (!success) success = southEast->insert(index[i]);
        index[i] = -1;
    }

    // Empty parent node
    size = 0;
    is_leaf = false;
}


// Build quadtree on dataset
void QuadTree::fill(int N)
{
    for (int i = 0; i < N; i++) insert(i);
}


// Checks whether the specified tree is correct
bool QuadTree::isCorrect()
{
    for (int n = 0; n < size; n++) {
        double* point = data + index[n] * QT_NO_DIMS;
        if (!boundary.containsPoint(point)) return false;
    }
    if (!is_leaf) return northWest->isCorrect() &&
                             northEast->isCorrect() &&
                             southWest->isCorrect() &&
                             southEast->isCorrect();
    else return true;
}


// Rebuilds a possibly incorrect tree (LAURENS: This function is not tested yet!)
void QuadTree::rebuildTree()
{
    for (int n = 0; n < size; n++) {

        // Check whether point is erroneous
        double* point = data + index[n] * QT_NO_DIMS;
        if (!boundary.containsPoint(point)) {

            // Remove erroneous point
            int rem_index = index[n];
            for (int m = n + 1; m < size; m++) index[m - 1] = index[m];
            index[size - 1] = -1;
            size--;

            // Update center-of-mass and counter in all parents
            bool done = false;
            QuadTree* node = this;
            while (!done) {
                for (int d = 0; d < QT_NO_DIMS; d++) {
                    node->center_of_mass[d] = ((double) node->cum_size * node->center_of_mass[d] - point[d]) / (double) (node->cum_size - 1);
                }
                node->cum_size--;
                if (node->getParent() == NULL) done = true;
                else node = node->getParent();
            }

            // Reinsert point in the root tree
            node->insert(rem_index);
        }
    }

    // Rebuild lower parts of the tree
    northWest->rebuildTree();
    northEast->rebuildTree();
    southWest->rebuildTree();
    southEast->rebuildTree();
}


// Build a list of all indices in quadtree
void QuadTree::getAllIndices(int* indices)
{
    getAllIndices(indices, 0);
}


// Build a list of all indices in quadtree
int QuadTree::getAllIndices(int* indices, int loc)
{

    // Gather indices in current quadrant
    for (int i = 0; i < size; i++) indices[loc + i] = index[i];
    loc += size;

    // Gather indices in children
    if (!is_leaf) {
        loc = northWest->getAllIndices(indices, loc);
        loc = northEast->getAllIndices(indices, loc);
        loc = southWest->getAllIndices(indices, loc);
        loc = southEast->getAllIndices(indices, loc);
    }
    return loc;
}


int QuadTree::getDepth() {
    if (is_leaf) return 1;
    return 1 + max(max(northWest->getDepth(),
                       northEast->getDepth()),
                   max(southWest->getDepth(),
                       southEast->getDepth()));

}


// Compute non-edge forces using Barnes-Hut algorithm
void QuadTree::computeNonEdgeForces(int point_index, double theta, double neg_f[], double* sum_Q, double buff[])
{

    // Make sure that we spend no time on empty nodes or self-interactions
    if (cum_size == 0 || (is_leaf && size == 1 && index[0] == point_index)) return;

    // Compute distance between point and center-of-mass
    double D = .0;
    int ind = point_index * QT_NO_DIMS;


    for (int d = 0; d < QT_NO_DIMS; d++) {
        buff[d]  = data[ind + d];
        buff[d] -= center_of_mass[d];
        D += buff[d] * buff[d];
    }

    // Check whether we can use this node as a "summary"
    if (is_leaf || max(boundary.hh, boundary.hw) / sqrt(D) < theta) {

        // Compute and add t-SNE force between point and current node
        double Q = 1.0 / (1.0 + D);
        *sum_Q += cum_size * Q;
        double mult = cum_size * Q * Q;
        for (int d = 0; d < QT_NO_DIMS; d++)
            neg_f[d] += mult * buff[d];
    }
    else {
        // Recursively apply Barnes-Hut to children
        northWest->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, buff);
        northEast->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, buff);
        southWest->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, buff);
        southEast->computeNonEdgeForces(point_index, theta, neg_f, sum_Q, buff);
    }
}


// Computes edge forces
void QuadTree::computeEdgeForces(int* row_P, int* col_P, double* val_P, int N, double* pos_f)
{

    // Loop over all edges in the graph
    int ind1, ind2;
    double D;
    double buff[QT_NO_DIMS];

    for (int n = 0; n < N; n++) {
        ind1 = n * QT_NO_DIMS;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Compute pairwise distance and Q-value
            D = .0;
            ind2 = col_P[i] * QT_NO_DIMS;
            for (int d = 0; d < QT_NO_DIMS; d++) {
                buff[d]  = data[ind1 + d];
                buff[d] -= data[ind2 + d];
                D += buff[d] * buff[d];

            }
            D = val_P[i] / (1.0 + D);

            // Sum positive force
            for (int d = 0; d < QT_NO_DIMS; d++) {
                pos_f[ind1 + d] += D * buff[d];
            }
        }
    }
}


// Print out tree
void QuadTree::print()
{
    if (cum_size == 0) {
        printf("Empty node\n");
        return;
    }

    if (is_leaf) {
        printf("Leaf node; data = [");
        for (int i = 0; i < size; i++) {
            double* point = data + index[i] * QT_NO_DIMS;
            for (int d = 0; d < QT_NO_DIMS; d++) printf("%f, ", point[d]);
            printf(" (index = %d)", index[i]);
            if (i < size - 1) printf("\n");
            else printf("]\n");
        }
    }
    else {
        printf("Intersection node with center-of-mass = [");
        for (int d = 0; d < QT_NO_DIMS; d++) printf("%f, ", center_of_mass[d]);
        printf("]; children are:\n");
        northEast->print();
        northWest->print();
        southEast->print();
        southWest->print();
    }
}

