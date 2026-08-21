"""Bit-equivalent scikit-image 0.26 noncompact watershed with caller-owned storage.

The flooding state machine, event fields, comparator, insertion order, neighbor order,
cost propagation, and watershed-line handling are transcribed from scikit-image 0.26.0.
Only the binary-heap representation differs: payloads are swapped directly in a fixed
caller-owned array instead of indirectly through a doubling pointer array.

See THIRD_PARTY_NOTICES.md for upstream source attribution and BSD-3-Clause terms.
"""

cimport cython
cimport numpy as cnp

cnp.import_array()


ctypedef cnp.int8_t DTYPE_BOOL_t


cdef struct Heapitem:
    cnp.int32_t cost_index
    cnp.int32_t age
    cnp.int32_t index
    cnp.int32_t source


cdef struct Heap:
    Py_ssize_t items
    Py_ssize_t space
    Heapitem *data
    const cnp.float64_t *image


cdef inline int smaller(Heapitem *a, Heapitem *b, Heap *heap) noexcept nogil:
    cdef cnp.float64_t a_value = heap.image[a.cost_index]
    cdef cnp.float64_t b_value = heap.image[b.cost_index]
    if a_value != b_value:
        return a_value < b_value
    return a.age < b.age


cdef inline cnp.int32_t propagated_cost_index(
    const cnp.float64_t *image,
    cnp.int32_t parent_cost_index,
    cnp.int32_t neighbor_index,
) noexcept nogil:
    if image[neighbor_index] < image[parent_cost_index]:
        return parent_cost_index
    return neighbor_index


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
            if smaller(&heap.data[left], &heap.data[i], heap):
                smallest = left
            if right < heap.items and smaller(
                &heap.data[right], &heap.data[smallest], heap
            ):
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
        if smaller(&heap.data[child], &heap.data[parent], heap):
            swap(parent, child, heap)
            child = parent
        else:
            break
    return 0


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


def heap_item_layout():
    """Return compiled Heapitem byte size and field offsets."""

    cdef Heapitem item
    cdef char *base = <char *>&item
    return {
        "size": sizeof(Heapitem),
        "cost_index": <Py_ssize_t>(<char *>&item.cost_index - base),
        "age": <Py_ssize_t>(<char *>&item.age - base),
        "index": <Py_ssize_t>(<char *>&item.index - base),
        "source": <Py_ssize_t>(<char *>&item.source - base),
    }


def event_smaller_for_test(
    const cnp.float64_t[::1] image,
    cnp.int32_t a_cost_index,
    cnp.int32_t a_age,
    cnp.int32_t b_cost_index,
    cnp.int32_t b_age,
):
    """Expose the exact cost/age comparator for frozen-binary verification."""

    cdef Heapitem a
    cdef Heapitem b
    cdef Heap heap
    if image.shape[0] == 0:
        raise ValueError("comparator test image must not be empty")
    if (
        a_cost_index < 0
        or b_cost_index < 0
        or a_cost_index >= image.shape[0]
        or b_cost_index >= image.shape[0]
    ):
        raise ValueError("comparator test cost index is out of range")
    heap.image = &image[0]
    a.cost_index = a_cost_index
    a.age = a_age
    b.cost_index = b_cost_index
    b.age = b_age
    return bool(smaller(&a, &b, &heap))


def propagated_cost_index_for_test(
    const cnp.float64_t[::1] image,
    cnp.int32_t parent_cost_index,
    cnp.int32_t neighbor_index,
):
    """Expose exact noncompact cost-origin propagation for verification."""

    if (
        parent_cost_index < 0
        or neighbor_index < 0
        or parent_cost_index >= image.shape[0]
        or neighbor_index >= image.shape[0]
    ):
        raise ValueError("cost-origin test index is out of range")
    return propagated_cost_index(&image[0], parent_cost_index, neighbor_index)


cdef inline void _validate_raveled_element_count(Py_ssize_t count) except *:
    if count > 2147483648:
        raise ValueError(
            "bounded watershed padded image exceeds the signed-int32 index range"
        )


def validate_raveled_element_count(Py_ssize_t count):
    """Expose the compiled flat-index boundary for allocation-free verification."""

    _validate_raveled_element_count(count)
    return count


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def watershed_raveled_bounded(
    const cnp.float64_t[::1] image,
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

    _validate_raveled_element_count(image.shape[0])
    if image.shape[0] == 0:
        raise ValueError("bounded watershed image must not be empty")
    if compactness != 0.0:
        raise ValueError("bounded watershed cost-index heap requires compactness == 0")
    if heap_storage.shape[0] < sizeof(Heapitem):
        raise ValueError("bounded watershed heap storage is too small")
    if heap_storage.shape[0] % sizeof(Heapitem):
        raise ValueError("bounded watershed heap storage has a partial item")

    heap.items = 0
    heap.space = heap_storage.shape[0] // sizeof(Heapitem)
    heap.data = <Heapitem *>&heap_storage[0]
    heap.image = &image[0]

    with nogil:
        for i in range(marker_locations.shape[0]):
            index = marker_locations[i]
            element.cost_index = <cnp.int32_t>index
            element.age = 0
            element.index = <cnp.int32_t>index
            element.source = <cnp.int32_t>index
            heappush(&heap, &element)

        while heap.items > 0:
            heappop(&heap, &element)

            if watershed_line:
                if output[element.index] and element.index != element.source:
                    continue
                if not _diff_neighbors(
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
                new_element.cost_index = <cnp.int32_t>neighbor_index
                if not watershed_line:
                    output[neighbor_index] = output[element.index]
                new_element.age = age
                new_element.index = <cnp.int32_t>neighbor_index
                new_element.source = element.source
                new_element.cost_index = propagated_cost_index(
                    heap.image,
                    element.cost_index,
                    new_element.cost_index,
                )
                heappush(&heap, &new_element)
