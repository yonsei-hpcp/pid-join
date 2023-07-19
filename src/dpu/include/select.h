#ifndef _SELECT_H_
#define _SELECT_H_

// Predicate (date)
bool predd(const uint32_t date, const uint32_t cmp_date, const uint32_t pred_operator)
{	
	if (pred_operator == 1) return date <= cmp_date;
	else if (pred_operator == 2) return date >= cmp_date;
	else if (pred_operator == 3) return date < cmp_date;
	else return date > cmp_date;
}

// Predicate (int/double)
bool predi(const double data, const double cmp_data, const uint32_t pred_operator)
{
	if (pred_operator == 1) return data <= cmp_data;
	else if (pred_operator == 2) return data >= cmp_data;
	else if (pred_operator == 3) return data < cmp_data;
	else return data > cmp_data;
}

bool predi64(const int64_t data, const int64_t cmp_data, const uint32_t pred_operator)
{
	if (pred_operator == 1) return data <= cmp_data;
	else if (pred_operator == 2) return data >= cmp_data;
	else if (pred_operator == 3) return data < cmp_data;
	else return data > cmp_data;
}

#define PREDINT64(data, cmp_data, pred_operator, result) \
{	\
	if (pred_operator == 0) result = (data == cmp_data);	\
	else if (pred_operator == 1) result = data <= cmp_data;	\
	else if (pred_operator == 2) result = data >= cmp_data;	\
	else if (pred_operator == 3) result = data < cmp_data;	\
	else result = data > cmp_data;	\
}	

bool preddb_btw(const double data, const double cmp_data1, const double cmp_data2)
{
	return ((cmp_data1 < data) && (data < cmp_data2));
}

bool predi_btw(const int64_t data, const int64_t cmp_data1, const int64_t cmp_data2)
{
	return ((cmp_data1 <= data) && (data <= cmp_data2));
}

bool predsm(const char* data, int32_t cmp_length, const int32_t cmp_str[30], const int32_t char_num[256])
{
	// Pointer for comparing / Position to start searching
	int ptr, start = 0;

	while (start < strlen(data) - cmp_length)
	{
		ptr = cmp_length - 1;

		while ((ptr >= 0) && (data[start + ptr] == cmp_str[ptr])) ptr--;

		// String Matched
		if (ptr < 0) return true;
		// String Unmatched
		else
		{
			if (ptr - char_num[(int)data[start + ptr]] <= 1) start++;
			else start += (ptr - char_num[(int)data[start + ptr]]);
		}
	}
	return false;
}

int preds(const char* str, const char* cmp_str)
{
	int chk = strncmp(str, cmp_str, strlen(str));

	if (chk == 0) return true;
	else return false;
}

#endif
