#include "tensor.h"
#include <stdint.h>
#include <assert.h>

// 1) Strides 2x3 -> {3,1}
static void test_strides(void) {
    Tensor t;
    size_t dims[2] = {2,3};
    tensor_create(&t, dims, 2);
    assert(t.strides[0] == 3 && t.strides[1] == 1);
    tensor_destroy(&t);
}

// 2) Owner create/destroy no leaks (basic)
static void test_owner(void) {
    Tensor t;
    size_t d[2]={100,100};
    assert(tensor_create(&t,d,2)==0);
    assert(tensor_is_contiguous(&t)==1);
    tensor_destroy(&t);
}

// 3) View copies shape but not data
static void test_view(void) {
    Tensor base; size_t d[2]={4,5};
    assert(tensor_create(&base,d,2)==0);
    size_t vd[2]={2,3}, vs[2]={5,1}; // top-left 2x3
    Tensor v;
    assert(tensor_view(&v,&base,vd,vs,2,0)==0);
    assert(v.owns_data==0 && v.owns_shape==1);
    tensor_destroy(&v);
    tensor_destroy(&base);
}

// 4) Overflow guard (huge dims should fail)
static void test_overflow(void) {
    Tensor t;
    size_t huge[2] = {SIZE_MAX/2, 5};
    assert(tensor_create(&t, huge, 2) == -1);
}
