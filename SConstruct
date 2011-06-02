from SCons.Node.FS import find_file

common = ["src/events.cpp", "src/model.cpp", "src/index.cpp"]
maxent = common + ["src/prior.cpp", "src/gistrainer.cpp", "src/lbfgstrainer.cpp",\
                   "src/maxent.cpp"]
nbayes = common + ["src/nbayes.cpp", "src/probs.cpp"]
env = Environment(CPPPATH = ["include"], LIBS=["boost_unit_test_framework-mt", "boost_program_options-mt", "boost_serialization-mt"])
env.Program("maxent", maxent)
#env.Program("nbayes", nbayes)

test_sources = Glob("test/*.cpp")
test_sources.extend(Glob("src/*.cpp"))
test_sources = filter(lambda f: f.name != "maxent.cpp", test_sources)
env.Program("tests", test_sources)
