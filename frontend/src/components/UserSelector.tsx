// frontend/src/components/UserSelector.tsx
import React, { useState, useCallback, forwardRef } from 'react';
import AsyncSelect from 'react-select/async';
import { SingleValue } from 'react-select';
import { searchUsers } from '../services/recommendationService';
import { User } from '../types';
import { debounce } from 'lodash';

interface UserOption {
  value: number;
  label: string;
}

interface UserSelectorProps {
  onUserSelect: (userId: number | null) => void;
  isLoading: boolean;
}

const UserSelector = forwardRef<any, UserSelectorProps>(({ onUserSelect, isLoading }, ref) => {
  const [selectedOption, setSelectedOption] = useState<SingleValue<UserOption>>(null);

  // Debounced function to fetch users
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const debouncedLoadUsers = useCallback(
    debounce(
      async ( inputValue: string, callback: (options: UserOption[]) => void ) => {
        if (!inputValue || inputValue.length < 1) {
          callback([]); return;
        }
        try {
          const users: User[] = await searchUsers(inputValue);
          const options = users.map(user => ({ value: user.student_id, label: String(user.student_id) }));
          callback(options);
        } catch (error) {
          console.error("Failed to load users via search:", error);
          callback([]);
        }
      },
      350
    ),
    []
  );

  const loadOptions = ( inputValue: string, callback: (options: UserOption[]) => void ) => {
    debouncedLoadUsers(inputValue, callback);
  };

  const handleChange = (selected: SingleValue<UserOption>) => {
    setSelectedOption(selected);
    onUserSelect(selected ? selected.value : null);
  };

  // Custom Styles for React Select (Black Theme)
  const customSelectStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: 'var(--color-surface)',
      borderColor: state.isFocused ? 'var(--color-primary)' : 'var(--color-border-color)',
      boxShadow: state.isFocused ? '0 0 0 1px var(--color-primary)' : 'none',
      color: 'var(--color-text-primary)',
      '&:hover': { borderColor: state.isFocused ? 'var(--color-primary)' : 'var(--color-border-color)', },
      borderRadius: '0.375rem',
      minHeight: '42px',
    }),
    menu: (provided: any) => ({
      ...provided,
      backgroundColor: 'var(--color-surface)',
      borderColor: 'var(--color-border-color)',
      borderWidth: '1px',
      borderRadius: '0.375rem',
      zIndex: 50,
    }),
    option: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: state.isSelected ? 'var(--color-primary)' : state.isFocused ? 'var(--color-border-color)' : 'var(--color-surface)',
      color: state.isSelected ? '#fff' : 'var(--color-text-primary)',
      '&:active': { backgroundColor: 'var(--color-primary)', },
      cursor: 'pointer',
    }),
    singleValue: (provided: any) => ({ ...provided, color: 'var(--color-text-primary)', }),
    input: (provided: any) => ({ ...provided, color: 'var(--color-text-primary)', margin: '0px', padding: '0px', }),
    placeholder: (provided: any) => ({ ...provided, color: 'var(--color-text-muted)', }),
    indicatorSeparator: (provided: any) => ({ ...provided, backgroundColor: 'var(--color-border-color)', }),
    dropdownIndicator: (provided: any, state: any) => ({
        ...provided, color: state.isFocused ? 'var(--color-primary)' : 'var(--color-text-muted)', '&:hover': { color: 'var(--color-primary)', }, cursor: 'pointer',
    }),
    clearIndicator: (provided: any) => ({ ...provided, color: 'var(--color-text-muted)', '&:hover': { color: 'var(--color-primary)', }, cursor: 'pointer', }),
    loadingIndicator: (provided: any) => ({ ...provided, color: 'var(--color-primary)', }),
    noOptionsMessage: (provided: any) => ({ ...provided, color: 'var(--color-text-muted)', }),
     valueContainer: (provided: any) => ({ ...provided, padding: '2px 8px', }),
  };

  return (
    // Container styling removed, handled by DemoPage
    <>
      <label htmlFor="userSelect" className="block text-sm font-medium text-text-secondary mb-2">
        Search Student ID:
      </label>
      <AsyncSelect<UserOption>
        ref={ref} // Forward the ref to AsyncSelect
        id="userSelect"
        cacheOptions
        loadOptions={loadOptions}
        defaultOptions
        value={selectedOption}
        onChange={handleChange}
        isDisabled={isLoading}
        placeholder={"Type to search..."}
        isClearable={true}
        noOptionsMessage={({ inputValue }) => !inputValue || inputValue.length < 1 ? 'Enter 1+ characters' : 'No matching users' }
        loadingMessage={() => 'Loading...'}
        styles={customSelectStyles}
        instanceId="user-async-select"
      />
    </>
  );
});

export default UserSelector;