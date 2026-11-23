//! A minimal FlatBuffers builder (back-to-front, like the reference implementation) used by the TensorFlow Lite exporter.
//!
//! Offsets are measured from the *end* of the buffer being built, so an object's offset is the number of bytes written
//! once it is complete. Tables are written without vtable deduplication.

pub(crate) struct Fb {
    /// The finished buffer reversed: `buf[len - 1]` is the first byte of the result.
    buf: Vec<u8>,
    minalign: usize,
    slots: Vec<(usize, usize)>, // (field index, offset from end of the field's data)
    table_start: usize,
}

pub(crate) type Off = usize;

impl Fb {
    pub fn new() -> Self {
        Self { buf: Vec::new(), minalign: 1, slots: Vec::new(), table_start: 0 }
    }

    fn head(&self) -> usize {
        self.buf.len()
    }

    /// Pads so that after `additional` more bytes the position is a multiple of `size`.
    fn prep(&mut self, size: usize, additional: usize) {
        self.minalign = self.minalign.max(size);
        let pad = (!(self.head() + additional)).wrapping_add(1) & (size - 1);
        self.buf.extend(std::iter::repeat_n(0u8, pad));
    }

    fn push_le(&mut self, le: &[u8]) {
        self.prep(le.len(), 0);
        self.buf.extend(le.iter().rev());
    }

    pub fn push_u32(&mut self, v: u32) {
        self.push_le(&v.to_le_bytes());
    }

    fn push_uoffset(&mut self, off: Off) {
        self.prep(4, 0);
        let rel = (self.head() + 4 - off) as u32;
        self.push_u32(rel);
    }

    pub fn create_string(&mut self, s: &str) -> Off {
        self.prep(4, s.len() + 1);
        self.buf.push(0);
        self.buf.extend(s.as_bytes().iter().rev());
        self.push_u32(s.len() as u32);
        self.head()
    }

    /// Vector of bytes (aligned to 16 so tensor data stays aligned for the runtime).
    pub fn create_bytes(&mut self, data: &[u8]) -> Off {
        self.prep(16, data.len());
        self.prep(4, data.len());
        self.buf.extend(data.iter().rev());
        self.push_u32(data.len() as u32);
        self.head()
    }

    pub fn create_i32_vec(&mut self, v: &[i32]) -> Off {
        self.prep(4, v.len() * 4);
        for x in v.iter().rev() {
            self.push_le(&x.to_le_bytes());
        }
        self.push_u32(v.len() as u32);
        self.head()
    }

    pub fn create_offset_vec(&mut self, v: &[Off]) -> Off {
        self.prep(4, v.len() * 4);
        for &x in v.iter().rev() {
            self.push_uoffset(x);
        }
        self.push_u32(v.len() as u32);
        self.head()
    }

    pub fn start_table(&mut self) {
        self.slots.clear();
        self.table_start = self.head();
    }

    pub fn add_u8(&mut self, field: usize, v: u8) {
        self.buf.push(v);
        self.slots.push((field, self.head()));
    }

    pub fn add_u32(&mut self, field: usize, v: u32) {
        self.push_le(&v.to_le_bytes());
        self.slots.push((field, self.head()));
    }

    pub fn add_i32(&mut self, field: usize, v: i32) {
        self.push_le(&v.to_le_bytes());
        self.slots.push((field, self.head()));
    }

    pub fn add_f32(&mut self, field: usize, v: f32) {
        self.push_le(&v.to_le_bytes());
        self.slots.push((field, self.head()));
    }

    pub fn add_offset(&mut self, field: usize, off: Off) {
        self.push_uoffset(off);
        self.slots.push((field, self.head()));
    }

    pub fn end_table(&mut self) -> Off {
        // soffset to the vtable, patched below
        self.push_le(&0i32.to_le_bytes());
        let table_loc = self.head();
        let max_field = self.slots.iter().map(|s| s.0).max().map_or(0, |m| m + 1);
        let mut entries = vec![0u16; max_field];
        for &(f, at) in &self.slots {
            entries[f] = (table_loc - at) as u16;
        }
        for &e in entries.iter().rev() {
            self.push_le(&e.to_le_bytes());
        }
        self.push_le(&((table_loc - self.table_start) as u16).to_le_bytes());
        self.push_le(&(((2 + max_field) * 2) as u16).to_le_bytes());
        let vt_loc = self.head();
        // the soffset (at distance `table_loc` from the end) = table position - vtable position = vt_loc - table_loc
        let bytes = ((vt_loc - table_loc) as i32).to_le_bytes();
        for (i, b) in bytes.iter().enumerate() {
            self.buf[table_loc - 1 - i] = *b; // the reversed buffer holds the least significant byte last
        }
        table_loc
    }

    /// Completes the buffer with the root table and a 4-byte file identifier.
    pub fn finish(mut self, root: Off, file_id: &[u8; 4]) -> Vec<u8> {
        self.prep(self.minalign.max(4), 8);
        self.buf.extend(file_id.iter().rev());
        self.push_uoffset(root);
        self.buf.reverse();
        self.buf
    }
}
