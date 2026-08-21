"""Bit-equivalent scikit-image 0.26 watershed core with caller-owned heap storage.

The flooding state machine, event fields, comparator, insertion order, neighbor order,
cost propagation, and watershed-line handling are transcribed from scikit-image 0.26.0.
Only the binary-heap representation differs: payloads are swapped directly in a fixed
caller-owned array instead of indirectly through a doubling pointer array.

See THIRD_PARTY_NOTICES.md for upstream source attribution and BSD-3-Clause terms.
"""

from libc.math cimport sqrt

cimport cython
cimport numpy as cnp

cnp.import_array()


ctypedef cnp.int8_t DTYPE_BOOL_t


cdef struct Heapitem:
    cnp.float64_t value
    cnp.int32_t age
    Py_ssize_t index
    Py_ssize_t source


cdef struct Heap:
    Py_ssize_t items
    Py_ssize_t space
    Heapitem *data


cdef inline int smaller(Heapitem *a, Heapitem *b) noexcept nogil:
    if a.value != b.value:
        return a.value < b.value
    return a.age < b.age


cdef inline void swap(Py_ssize_t a, Py_ssize_t b, Heap *heap) noexcept nogil:
    cdef Heapitem temporary = heap.data[a]
    heap.data[a] = heap.data[b]
    heap.data[b] = temporary


cdef inline void heappop(Heap *heap, Heapitem *destination) noexcept nogil:
    cdef Py_ssize_t i, smallest, left, right

    destination[0] = heap.data[0]
    heap.items -= 1
    if heap.items == 0:
        return

    heap.data[0] = heap.data[heap.items]
    i = 0
    smallest = i
    while True:
        left = i * 2 + 1
        right = i * 2 + 2
        if left < heap.items:
            if smaller(&heap.data[left], &heap.data[i]):
                smallest = left
            if right < heap.items and smaller(&heap.data[right], &heap.data[smallest]):
                smallest = right
        else:
            break
        if smallest == i:
            break
        swap(i, smallest, heap)
        i = smallest


cdef inline int heappush(Heap *heap, Heapitem *new_element) except -1 nogil:
    cdef Py_ssize_t child = heap.items
    cdef Py_ssize_t parent

    if child == heap.space:
        with gil:
            raise MemoryError("bounded watershed heap capacity exhausted")

    heap.data[child] = new_element[0]
    heap.items += 1
    while child > 0:
        parent = (child + 1) // 2 - 1
        if smaller(&heap.data[child], &heap.data[parent]):
            swap(parent, child, heap)
            child = parent
        else:
            break
    return 0


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef inline cnp.float64_t _euclid_dist(
    Py_ssize_t point0,
    Py_ssize_t point1,
    cnp.intp_t[::1] strides,
) noexcept nogil:
    cdef cnp.float64_t result = 0
    cdef cnp.float64_t current = 0
    cdef Py_ssize_t i
    for i in range(strides.shape[0]):
        current = (point0 // strides[i]) - (point1 // strides[i])
        result += current * current
        point0 = point0 % strides[i]
        point1 = point1 % strides[i]
    return sqrt(result)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline DTYPE_BOOL_t _diff_neighbors(
    cnp.int32_t[::1] output,
    cnp.intp_t[::1] structure,
    DTYPE_BOOL_t[::1] mask,
    Py_ssize_t index,
    cnp.int32_t label,
) noexcept nogil:
    cdef Py_ssize_t i, neighbor_index
    cdef cnp.int32_t neighbor_label
    cdef Py_ssize_t neighbor_count = structure.shape[0]

    if not mask[index]:
        return True
    for i in range(neighbor_count):
        neighbor_index = structure[i] + index
        if mask[neighbor_index]:
            neighbor_label = output[neighbor_index]
            if neighbor_label and neighbor_label != label:
                mask[index] = False
                return True
    return False


def heap_item_size():
    """Return the compiled Heapitem size used to size the backing file."""

    return sizeof(Heapitem)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def watershed_raveled_bounded(
    cnp.float64_t[::1] image,
    cnp.intp_t[::1] marker_locations,
    cnp.intp_t[::1] structure,
    DTYPE_BOOL_t[::1] mask,
    cnp.intp_t[::1] strides,
    cnp.float64_t compactness,
    cnp.int32_t[::1] output,
    DTYPE_BOOL_t watershed_line,
    cnp.uint8_t[::1] heap_storage,
):
    """Run the pinned watershed event stream in caller-owned heap storage."""

    cdef Heapitem element
    cdef Heapitem new_element
    cdef Heap heap
    cdef Py_ssize_t neighbor_count = structure.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t age = 1
    cdef Py_ssize_t index = 0
    cdef Py_ssize_t neighbor_index = 0
    cdef DTYPE_BOOL_t compact = compactness > 0

    if heap_storage.shape[0] < sizeof(Heapitem):
        raise ValueError("bounded watershed heap storage is too small")
    if heap_storage.shape[0] % sizeof(Heapitem):
        raise ValueError("bounded watershed heap storage has a partial item")

    heap.items = 0
    heap.space = heap_storage.shape[0] // sizeof(Heapitem)
    heap.data = <Heapitem *>&heap_storage[0]

    with nogil:
        for i in range(marker_locations.shape[0]):
            index = marker_locations[i]
            element.value = image[index]
            element.age = 0
            element.index = index
            element.source = index
            heappush(&heap, &element)

        while heap.items > 0:
            heappop(&heap, &element)

            if compact or watershed_line:
                if output[element.index] and element.index != element.source:
                    continue
                if compact or not _diff_neighbors(
                    output,
                    structure,
                    mask,
                    element.index,
                    output[element.source],
                ):
                    output[element.index] = output[element.source]

            for i in range(neighbor_count):
                neighbor_index = structure[i] + element.index
                if not mask[neighbor_index]:
                    continue
                if output[neighbor_index]:
                    continue

                age += 1
                new_element.value = image[neighbor_index]
                if compact:
                    new_element.value += compactness * _euclid_dist(
                        neighbor_index,
                        element.source,
                        strides,
                    )
                elif not watershed_line:
                    output[neighbor_index] = output[element.index]
                new_element.age = age
                new_element.index = neighbor_index
                new_element.source = element.source
                if new_element.value < element.value:
                    new_element.value = element.value
                heappush(&heap, &new_element)
