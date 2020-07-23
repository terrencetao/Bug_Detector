class Person {
};


class Room {
public:
    void add_person(Person person)
    {
        // do stuff
    }

private:
    Person* people_in_room;
    int t[4];
    vector<int> vt(5,3);
};


template <class T, int N>
class Bag<T, N> {
};


int main()
{
    Person* p = new Person();
    Bag<Person, 42> bagofpersons;

