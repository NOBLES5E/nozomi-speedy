use crate::private::write_length;
use std::hash::{BuildHasher, Hash};
use std::mem;
use {
    crate::{Context, Readable, Reader, Writable, Writer},
    hashbrown::HashMap,
};

impl<'a, C, K, V, S> Readable<'a, C> for HashMap<K, V, S>
where
    C: Context,
    K: Readable<'a, C> + Eq + Hash,
    V: Readable<'a, C>,
    S: BuildHasher + Default,
{
    #[inline]
    fn read_from<R: Reader<'a, C>>(reader: &mut R) -> Result<Self, C::Error> {
        let length = crate::private::read_length(reader)?;
        reader.read_collection(length)
    }

    #[inline]
    fn minimum_bytes_needed() -> usize {
        4
    }
}

impl<C, K, V, S> Writable<C> for HashMap<K, V, S>
where
    C: Context,
    K: Writable<C>,
    V: Writable<C>,
{
    #[inline]
    fn write_to<W: ?Sized + Writer<C>>(&self, writer: &mut W) -> Result<(), C::Error> {
        write_length(self.len(), writer)?;
        writer.write_collection(self.iter())
    }

    #[inline]
    fn bytes_needed(&self) -> Result<usize, C::Error> {
        unsafe_is_length!(self.len());

        let mut count = mem::size_of::<u32>();
        for (key, value) in self {
            count += key.bytes_needed()? + value.bytes_needed()?;
        }

        Ok(count)
    }
}
