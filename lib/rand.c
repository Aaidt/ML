#include <stdint.h>
#include <stdlib.h>

#include <math.h>

typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;

typedef struct {
  u64 state;
  u64 inc;
} prng_state;

void prng_seed_r(prng_state *rng, u64 initstate, u64 initseq);
void prng_seed(u64 initstate, u64 initseq);

u32 prng_rand_r(prng_state *rng);
u32 prng_rand(void);

int main(void) { return 0; }
