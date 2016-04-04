
//Fujitsu K
#ifndef GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_FUJITSU_SPARC_H_
#define GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_FUJITSU_SPARC_H_

//#include <stl/_threads.h>
//#include <stdint.h>

namespace google
{
	namespace protobuf
	{
		namespace internal
		{
			inline Atomic32 NoBarrier_CompareAndSwap(volatile Atomic32* ptr,
													 Atomic32 old_value,
													 Atomic32 new_value) 
			{
				register volatile int prev;
				__asm__ __volatile__ ("\t! atomic_cas_32\n\t"
									"mov      %4, %0\n\t"
									"cas      [%2], %3, %0\n"
									: "=&r" (prev), "=m" (*ptr)
									: "r" (ptr), "r" (old_value), "r" (new_value), "m" (*ptr));
				return prev;
				//__atomic_compare_exchange_n(ptr, &old_value, new_value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);//gcc
				//return old_value;
			}

			inline Atomic32 NoBarrier_AtomicExchange(volatile Atomic32* ptr,
													 Atomic32 new_value)
			{
				register volatile int old_v, new_v;
				__asm__ __volatile__ ("\t! atomic_swap_32\n\t"
									"ld       [%3], %0\n"
									"1:\n\t"
									"mov      %4, %1\n\t"
									"cas      [%3], %0, %1\n\t"
									"cmp      %0, %1\n\t"
									"bne,a,pn %%icc, 1b\n\t"
									" mov     %1, %0\n\t"
									"mov      %4, %1\n"
									: "=&r" (old_v), "=&r" (new_v), "=m" (*ptr)
									: "r" (ptr), "r" (new_value), "m" (*ptr)
									: "cc");
				/* if (old_value) { } */
				if (new_v) { }
				return old_v;
				//return __atomic_exchange_n(ptr, new_value, __ATOMIC_RELAXED);
			}

			inline Atomic32 NoBarrier_AtomicIncrement(volatile Atomic32* ptr,
													  Atomic32 increment)
			{
				register volatile int old_value, new_value;
				__asm__ __volatile__ ("\t! atomic_add_32\n\t"
									"ld       [%3], %0\n"
									"1:\n\t"
									"add      %4, %0, %1\n\t"
									"cas      [%3], %0, %1\n\t"
									"cmp      %0, %1\n\t"
									"bne,a,pn %%icc, 1b\n\t"
									" mov     %1, %0\n\t"
									"add      %4, %0, %1\n"
									: "=&r" (old_value), "=&r" (new_value), "=m" (*ptr)
									: "r" (ptr), "ir" (increment), "m" (*ptr)
									: "cc");
				if (old_value) { }
				if (new_value) { }
				return *ptr;
				//return __atomic_add_fetch(ptr, increment, __ATOMIC_RELAXED);
			}

			inline Atomic32 Barrier_AtomicIncrement(volatile Atomic32* ptr,
													Atomic32 increment)
			{
				register volatile int old_value, new_value;
				__asm__ __volatile__ ("\t! atomic_add_32\n\t"
									"ld       [%3], %0\n"
									"1:\n\t"
									"add      %4, %0, %1\n\t"
									"cas      [%3], %0, %1\n\t"
									"cmp      %0, %1\n\t"
									"bne,a,pn %%icc, 1b\n\t"
									" mov     %1, %0\n\t"
									"add      %4, %0, %1\n"
									: "=&r" (old_value), "=&r" (new_value), "=m" (*ptr)
									: "r" (ptr), "ir" (increment), "m" (*ptr)
									: "cc");
				if (old_value) { }
				if (new_value) { }
				MemoryBarrier();
				return *ptr;
				//return __atomic_add_fetch(ptr, increment, __ATOMIC_SEQ_CST);
			}

			inline Atomic32 Acquire_CompareAndSwap(volatile Atomic32* ptr,
												   Atomic32 old_value,
												   Atomic32 new_value)
			{
				register volatile int prev;
				__asm__ __volatile__ ("\t! atomic_cas_32\n\t"
									"mov      %4, %0\n\t"
									"cas      [%2], %3, %0\n"
									: "=&r" (prev), "=m" (*ptr)
									: "r" (ptr), "r" (old_value), "r" (new_value), "m" (*ptr));
				MemoryBarrier();
				return prev;
				//__atomic_compare_exchange_n(ptr, &old_value, new_value, true, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
				//return old_value;
			}

			inline Atomic32 Release_CompareAndSwap(volatile Atomic32* ptr,
												   Atomic32 old_value,
												   Atomic32 new_value)
			{
				register volatile int prev;
				__asm__ __volatile__ ("\t! atomic_cas_32\n\t"
									"mov      %4, %0\n\t"
									"cas      [%2], %3, %0\n"
									: "=&r" (prev), "=m" (*ptr)
									: "r" (ptr), "r" (old_value), "r" (new_value), "m" (*ptr));

				return prev;
				//__atomic_compare_exchange_n(ptr, &old_value, new_value, true, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
				//return old_value;
			}

			inline void MemoryBarrier()
			{
				__asm__ __volatile__ ("\t! membar_consumer\n\t"
									"membar   #LoadLoad\n");
				return;
			}

			inline void NoBarrier_Store(volatile Atomic32* ptr, Atomic32 value)
			{
				*ptr = value;/* Aligned loads and stores are atomic on sparc. */
				//__atomic_store_n(ptr, value, __ATOMIC_RELAXED);
			}

			inline void Acquire_Store(volatile Atomic32* ptr, Atomic32 value)
			{
				*ptr = value;/* Aligned loads and stores are atomic on sparc. */
				MemoryBarrier();
			}

			inline void Release_Store(volatile Atomic32* ptr, Atomic32 value)
			{
				*ptr = value;/* Aligned loads and stores are atomic on sparc. */
			}

			inline Atomic32 NoBarrier_Load(volatile const Atomic32* ptr)
			{
				return *ptr;
				//return __atomic_load_n(ptr, __ATOMIC_RELAXED);
			}

			inline Atomic32 Acquire_Load(volatile const Atomic32* ptr)
			{
				return *ptr;
				//return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
			}

			inline Atomic32 Release_Load(volatile const Atomic32* ptr)
			{
				return *ptr;
				//return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
			}

			#ifdef __LP64__

				inline void  NoBarrier_Store(volatile Atomic64* ptr, Atomic64 value)
				{
					*ptr = value;
					//__atomic_store_n(ptr, value, __ATOMIC_RELEASE);
				}

				inline void Release_Store(volatile Atomic64* ptr, Atomic64 value)
				{
					*ptr = value;
					//__atomic_store_n(ptr, value, __ATOMIC_RELEASE);
				}

				inline Atomic64 NoBarrier_Load(volatile const Atomic64* ptr)
				{
					return *ptr;
					//return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
				}

				inline Atomic64 Acquire_Load(volatile const Atomic64* ptr)
				{
					return *ptr;
					//return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
				}

				inline Atomic64 Acquire_CompareAndSwap(volatile Atomic64* ptr,
													   Atomic64 old_value,
													   Atomic64 new_value)
				{
					#ifdef	notdef2
						void *prev;
					#else	/* notdef2 */
						register volatile uint64_t prev;
					#endif	/* notdef2 */
						__asm__ __volatile__ ("\t! atomic_cas_64\n\t"
						"mov      %4, %0\n\t"
						"casx     [%2], %3, %0\n"
						: "=&r" (prev), "=m" (*ptr)
						: "r" (ptr), "r" (&old_value), "r" (&new_value), "m" (*ptr));
						MemoryBarrier();
					#ifdef	notdef2
						return prev;
					#else	/* notdef2 */
						return (Atomic64)prev;
					#endif	/* notdef2 */
					
					//__atomic_compare_exchange_n(ptr, &old_value, new_value, true, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);
					//return old_value;
				}

				inline Atomic64 NoBarrier_CompareAndSwap(volatile Atomic64* ptr,
														 Atomic64 old_value,
														 Atomic64 new_value)
				{
					#ifdef	notdef2
						void *prev;
					#else	/* notdef2 */
						register volatile uint64_t prev;
					#endif	/* notdef2 */
						__asm__ __volatile__ ("\t! atomic_cas_64\n\t"
						"mov      %4, %0\n\t"
						"casx     [%2], %3, %0\n"
						: "=&r" (prev), "=m" (*ptr)
						: "r" (ptr), "r" (&old_value), "r" (&new_value), "m" (*ptr));
					#ifdef	notdef2
						return prev;
					#else	/* notdef2 */
						return (Atomic64)prev;
					#endif	/* notdef2 */
					//__atomic_compare_exchange_n(ptr, &old_value, new_value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
					//return old_value;
				}
			inline Atomic64 NoBarrier_AtomicExchange(volatile Atomic64* ptr,
													 Atomic64 new_value)
			{
				register volatile int old_v, new_v;
				__asm__ __volatile__ ("\t! atomic_swap_64\n\t"
									"ldx       [%3], %0\n"
									"1:\n\t"
									"mov      %4, %1\n\t"
									"casx      [%3], %0, %1\n\t"
									"cmp      %0, %1\n\t"
									"bne,a,pn %%icc, 1b\n\t"
									" mov     %1, %0\n\t"
									"mov      %4, %1\n"
									: "=&r" (old_v), "=&r" (new_v), "=m" (*ptr)
									: "r" (ptr), "r" (new_value), "m" (*ptr)
									: "cc");
				/* if (old_value) { } */
				if (new_v) { }
				return old_v;
				//return __atomic_exchange_n(ptr, new_value, __ATOMIC_RELAXED);
			}
			inline Atomic64 NoBarrier_AtomicIncrement(volatile Atomic64* ptr,
													Atomic64 increment)
			{
				register volatile int old_value, new_value;
				__asm__ __volatile__ ("\t! atomic_add_64\n\t"
									"ldx       [%3], %0\n"
									"1:\n\t"
									"add      %4, %0, %1\n\t"
									"casx      [%3], %0, %1\n\t"
									"cmp      %0, %1\n\t"
									"bne,a,pn %%icc, 1b\n\t"
									" mov     %1, %0\n\t"
									"add      %4, %0, %1\n"
									: "=&r" (old_value), "=&r" (new_value), "=m" (*ptr)
									: "r" (ptr), "ir" (increment), "m" (*ptr)
									: "cc");
				if (old_value) { }
				if (new_value) { }
				MemoryBarrier();
				return *ptr;
				//return __atomic_add_fetch(ptr, increment, __ATOMIC_SEQ_CST);
			}

			#endif // defined(__LP64__)

		}  // namespace internal
	}  // namespace protobuf
}  // namespace google

#endif  // GOOGLE_PROTOBUF_ATOMICOPS_INTERNALS_FUJITSU_SPARC_H_
