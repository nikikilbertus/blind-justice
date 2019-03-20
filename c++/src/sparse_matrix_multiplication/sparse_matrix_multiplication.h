/*  We want to implement the following protocol.
    Public inputs:
    K_A: Number of words in all of A's documents
    K_B: Number of words in B's document
    V: Vocabulary
    D: Number of documents A holds

    Private inputs:
    A: in_A: Array[K_A] of of words
       values_A: Array[K_A] of arrays[D] of counts
    B: in_B: Array[K_B] of pairs (k, v) where k is a word and v is an index into the following
       values_B: Array[K_A + K_B] of permuted values, where K_B are actual counts and K_A are zeroes
       zeroes_B: Array[K_A] holding indices to zeroes in the above

    Protocol:
    1. B homomorphically encrypts values_B and sends it to A
    2. A and B run a garbled circuit that implements the following algorithm:

        L: Array[K_A + K_B] of triples (k, party, v)
        for i in range(K_A):
            L[i] = (in_A[i], 0, i)
        for i in range(K_B):
            L[K_A + i] = (in_B[i].k, 1, in_B[i].v)
        sort L by fields (k, party) # ensures A's value comes always first if keys are equal
        M: Array[K_A + K_B - 1] of tuples (v_A, v_B, valid, equal)
        for i in range(len(L) - 1):
            valid = (L[i].party == 0) # holds for exactly K_A elements
            equal = (L[i].k == L[i+1].k) # holds for at most min(K_A, K_B) elements
            M[i] = (L[i].v, L[i+1].v, valid, equal)
        sort M by fields (valid, r) where r is chosen randomly # "true" comes before "false"
        N: Array[K_A] of pairs (v_A, v_B)
        for i in range(K_A):
            if M[i].equal:
                N[i] = M[i]
            else:
                N[i] = (M[i].v_A, zeroes_B[i])
        output N to player A

    3. For each (v_A, v_B) in N, player A homomorphically multiplies their vector
       values_A[v_A] with the encrypted scalar values_B[v_B], sums the results, and
       masks the resulting vector of size D
*/


#pragma once

typedef unsigned short index_t; // type for indexing into values arrays
typedef unsigned int word_t;  // type for indexing into the vocabulary

typedef struct {
    index_t v_A;
    index_t v_B;
} index_index_pair; // element type of the returned vector

typedef struct {
    word_t k;
    index_t v;
} word_index_pair; // element type of B's input vector

typedef struct {
    size_t l;
    size_t m;
    size_t n;
    size_t k_A; // number of non-zero columns in A's l x m matrix
    size_t k_B; // number of non-zero elements in each row of B's m x n matrix
    word_t *in_A; // Array[K_A] of of words
    word_index_pair *in_B; // Array[K_B] of pairs (k, v) where k is a word and v is an index into values_B
    index_t *zeroes_B; // Array[K_A] holding indices to zeroes in values_B
    index_index_pair *result; // Array[K_A] of pairs (v_A, v_B)
} sparse_matrix_multiplication_args;


void sparse_matrix_multiplication(void *);
