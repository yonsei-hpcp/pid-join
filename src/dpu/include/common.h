#ifndef _COMMON_H_
#define _COMMON_H_


// Structures used by both the host and the dpu to communicate information
int32_t log2_manual(uint32_t value)
{
    for (int i = 0; ; i++)
    {
        if (((value >> i) & 0x1) == 0x1)
        {
            return i;
        }
    }
}

void quickSort(int64_t *arr, int64_t *payload, uint8_t length, uint8_t *stack)
{
    int subArray = 0;

    uint8_t ptr = 0;
    stack[ptr++] = 0;
    stack[ptr++] = length -1;

    do
    {
        ptr--;
        uint8_t right = stack[ptr];
        ptr--;
        uint8_t left = stack[ptr];
        --subArray;
        do
        {
            uint8_t _left = left;
            uint8_t _right = right;
            int64_t pivot = arr[(left + right) / 2];
            do
            {
                while (pivot < arr[_right])
                {
                    _right--;
                }
                while (pivot > arr[_left])
                {
                    _left++;
                }
                if (_left <= _right)
                {
                    if (_left != _right)
                    {
                        int64_t temp = arr[_left];
                        arr[_left] = arr[_right];
                        arr[_right] = temp;

                        temp = payload[_left];
                        payload[_left] = payload[_right];
                        payload[_right] = temp;
                    }
                    _right--;
                    _left++;
                }
            } while (_right >= _left);
            if (_left < right)
            {
                ++subArray;

                stack[ptr++] = _left;
                stack[ptr++] = right;
            }
            right = _right;
        } while (left < right);
    } while (subArray > -1);
}

#endif
