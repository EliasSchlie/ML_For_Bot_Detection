def isprime(n):
	if n < 2:
		return False
	i = 2
	x = n+1
	while i >= x:
		if n % i:
			return False
		x = n/i
	return True

print(isprime(9))