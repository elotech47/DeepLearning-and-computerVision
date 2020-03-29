# import the necessary packages
import argparse
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of the user")
ap.add_argument("-a", "--age", required=True,
	help="age of the user")
args = vars(ap.parse_args())
# display a friendly message to the user
print("Hi there {}, it's nice to meet you!".format(args["name"]))
UserAge = args["age"]
if int(UserAge) < 18:
    print("{}, you're not yet an adult".format(args["name"]))
else:
    print("{}, you're an adult".format(args["name"]))