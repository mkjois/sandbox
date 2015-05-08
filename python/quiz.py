from functools import reduce
import timeit
import urllib.request

def function(data):
    n, s, ss = reduce(lambda a, b: map(sum, zip(a,b)),
                      [(1, x, x*x) for x in data])
    return (ss - s*s/n) / n

def variance(data):
  def helper(nums):
    if len(nums) == 0:
      return 0, 0, 0
    if len(nums) == 1:
      datum = nums[0]
      return 1, datum, datum * datum
    l = len(nums)
    n1, u1, m1 = helper(nums[:int(l/2)])
    n2, u2, m2 = helper(nums[int(l/2):])
    n = n1 + n2
    u = (n1*u1 + n2*u2) / n
    m = (n1*m1 + n2*m2) / n
    return n, u, m
  n, first_moment, second_moment = helper(data)
  return second_moment - first_moment * first_moment

def memthrift(data):
  n, s, ss = len(data), 0, 0
  for x in data:
    s += x
    ss += x*x
  return (ss - s*s/n) / n

def best(data):
  def helper(a, b):
    if b-a == 1:
      datum = data[a]
      return 1, datum, datum * datum
    mid = int((b+a)/2)
    n1, u1, m1 = helper(a, mid)
    n2, u2, m2 = helper(mid, b)
    n = n1 + n2
    u = (n1*u1 + n2*u2) / n
    m = (n1*m1 + n2*m2) / n
    return n, u, m
  n, first_moment, second_moment = helper(0, len(data))
  return second_moment - first_moment * first_moment


if __name__ == "__main__":
  """
  # create a password manager
  password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()

  # Add the username and password.
  # If we knew the realm, we could use it instead of None.
  top_level_url = "http://remeeting.com/quiz/"
  password_mgr.add_password(None, top_level_url, "manny", "asdfmoney")

  handler = urllib.request.HTTPBasicAuthHandler(password_mgr)

  # create "opener" (OpenerDirector instance)
  opener = urllib.request.build_opener(handler)

  # use the opener to fetch a URL
  fav_number = 0
  for i in range(100,1000):
    with opener.open("http://remeeting.com/quiz/python_script.cgi?number=" + str(i)) as f:
      if not b"Sorry" in f.read():
        fav_number = i
        break
  print("Favorite number is %d" % fav_number)

  # Install the opener.
  # Now all calls to urllib.request.urlopen use our opener.
  urllib.request.install_opener(opener)
  """

  size, trials = 50000, 10
  f = lambda: function(range(size))
  v = lambda: variance(range(size))
  m = lambda: memthrift(range(size))
  b = lambda: best(range(size))
  t1 = timeit.timeit(stmt="f()", setup="from __main__ import f", number=trials)
  t2 = timeit.timeit(stmt="v()", setup="from __main__ import v", number=trials)
  t3 = timeit.timeit(stmt="m()", setup="from __main__ import m", number=trials)
  t4 = timeit.timeit(stmt="b()", setup="from __main__ import b", number=trials)
  print("Running 'function' on list of size %d for %d trials: %f sec" % (size, trials, t1))
  print("Running 'variance' on list of size %d for %d trials: %f sec" % (size, trials, t2))
  print("Running 'memthrift' on list of size %d for %d trials: %f sec" % (size, trials, t3))
  print("Running 'best' on list of size %d for %d trials: %f sec" % (size, trials, t4))
