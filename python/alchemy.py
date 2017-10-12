import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref, subqueryload, joinedload, contains_eager
from sqlalchemy.sql import exists
from sqlalchemy import Column, ForeignKey, Integer, String, text, func
print(sa.__version__)

Base = declarative_base()
Session = sessionmaker()

class Person(Base):
  __tablename__ = "person"

  id = Column(Integer, primary_key=True)
  fname = Column(String(20))
  lname = Column(String(20))
  pw = Column(String(15), nullable=False)
  emails = relationship("Email", backref="person", cascade="all, delete, delete-orphan")

  def __repr__(self):
    return "<User(fname='%s', lname='%s', pw='%s')>" % (self.fname, self.lname, self.pw)

class Email(Base):
  __tablename__ = "email"
  id = Column(Integer, primary_key=True)
  email = Column(String, nullable=False)
  person_id = Column(Integer, ForeignKey("person.id"))
  #person = relationship("Person", backref=backref("emails", order_by=id)) # Person.emails is ordered by Email.id

  def __repr__(self):
    return "<Email(email='%s')>" % self.email

manny = Person(fname="manny", lname="jois", pw="mj")
shrayus = Person(fname="shrayus", lname="gupta", pw="sg")
ishan = Person(fname="ishan", lname="shah", pw="is")
junseok = Person(fname="junseok", lname="lee", pw="jl")
kyle = Person(fname="kyle", lname="hirai", pw="kh")

manny.emails = [Email(email="m.k.jois@gmail.com"),
                Email(email="m.k.jois@berkeley.edu")]
junseok.emails = [Email(email="lee.junseok@berkeley.edu")]
kyle.emails = [Email(email="khirai94@yahoo.com")]

engine = sa.create_engine("postgresql+psycopg2://username:password@sample-1.ci7vq8x4kd8z.us-west-2.rds.amazonaws.com:5432/sample1")
Base.metadata.create_all(engine)
Session.configure(bind=engine)

s1 = Session()
s1.add_all([manny, shrayus, ishan, junseok, kyle])
s1.commit()

# properties of relationships
print(manny.emails[1])
print(manny.emails[1].person)

# example queries and operators
for f, l in s1.query(Person.fname, Person.lname).order_by(Person.id):
  print("%s %s" % (f, l))
for row in s1.query(Person.fname, Person.lname.label("surname")).filter(Person.lname.like("%h%")).all():
  print("%s %s" % (row.fname, row.surname))
for row in s1.query(Person.pw).filter(text("id < :value")).params(value=4).order_by(Person.id):
  print(row.pw)
num_users = s1.query(func.count(Person.id)).scalar()
print(num_users)

# queries involving joins
for p, e in s1.query(Person, Email)\
                   .filter(Person.id == Email.person_id)\
                   .filter(Email.email == "m.k.jois@berkeley.edu"):
  print(p, e)
for person in s1.query(Person).join(Email)\
                .filter(Email.email.like("%berkeley.edu")):
  print(person)
existensial = exists().where(Email.person_id == Person.id)
for name, in s1.query(Person.fname).filter(existensial):
  print(name)
for email in s1.query(Email).filter(Email.person.has(Person.fname == "manny")):
  print(email)

# optimized loading
ky = s1.query(Person).options(subqueryload(Person.emails)).filter_by(fname="kyle").first()
print(ky)
print(ky.emails)
print(s1.query(Person).join(Email).filter(Person.fname == "manny").first())
print(s1.query(Person).options(joinedload(Person.emails)).filter(Person.fname == "manny").first().emails)

# deleting
mj = s1.query(Person).get(1) # load by primary key
del mj.emails[0] # deletes first email address
print(s1.query(Email).filter(Email.email.like("m%")).count())
print(mj.emails[0])
s1.delete(mj) # cascade deletes associated emails too
s1.commit()
print(s1.query(Email).filter(Email.email.like("m%")).count())

s1.close()
Base.metadata.drop_all(engine)
